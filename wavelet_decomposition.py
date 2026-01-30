import pydicom
import pywt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def subband_decomposition(img):
    """子带编码：使用小波变换 (无冗余)"""
    coeffs = pywt.wavedec2(img, 'haar', level=1)
    LL, (LH, HL, HH) = coeffs
    # LL, LH, HL, HH 尺寸均为原图 1/4，总数 = 1
    return [LL, LH, HL, HH]

def pyramidal_decomposition(img):
    """金字塔编码：拉普拉斯金字塔 (有冗余)"""
    # 1. 高斯模糊下采样
    lower = cv2.pyrDown(img)
    # 2. 上采样回原尺寸
    upper = cv2.pyrUp(lower, dstsize=(img.shape[1], img.shape[0]))
    # 3. 计算残差 (Laplacian)
    laplacian = cv2.subtract(img, upper)
    return [lower, laplacian]

# --- 交互显示逻辑 ---
def display_coding_comparison(csv_path, num_files=127):
    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    files = df['FullPath'].head(num_files).tolist()
    
    fig = plt.figure(figsize=(20, 10))
    current_idx = [0]

    def show_next():
        if current_idx[0] >= len(files): return
        f = files[current_idx[0]]
        try:
            plt.clf()
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
            # 1. 子带分解
            sb_parts = subband_decomposition(img)
            # 2. 金字塔分解
            py_parts = pyramidal_decomposition(img)
            
            # --- 绘图布局 ---
            # 子带编码展示 (4个子带)
            titles_sb = ['LL (Approx)', 'LH (Horiz)', 'HL (Vert)', 'HH (Diag)']
            for i in range(4):
                ax = fig.add_subplot(2, 4, i+1)
                ax.imshow(sb_parts[i], cmap='gray')
                ax.set_title(f"Subband: {titles_sb[i]}\nSize: {sb_parts[i].shape}")
                ax.axis('off')
            
            # 金字塔编码展示 (G1 和 L0)
            ax_p1 = fig.add_subplot(2, 4, 5)
            ax_p1.imshow(py_parts[0], cmap='gray')
            ax_p1.set_title(f"Pyramid: G1 (Downsampled)\nSize: {py_parts[0].shape}")
            ax_p1.axis('off')
            
            ax_p2 = fig.add_subplot(2, 4, 6)
            # 增强残差显示
            lap_vis = cv2.normalize(py_parts[1], None, 0, 255, cv2.NORM_MINMAX)
            ax_p2.imshow(lap_vis, cmap='magma')
            ax_p2.set_title(f"Pyramid: L0 (Residual)\nSize: {py_parts[1].shape}")
            ax_p2.axis('off')
            
            # 数据量统计说明
            sb_count = sum([p.size for p in sb_parts])
            py_count = sum([p.size for p in py_parts])
            info_text = (f"Original Pixels: {img.size}\n"
                         f"Subband Total: {sb_count} (Redundancy: 0%)\n"
                         f"Pyramid Total: {py_count} (Redundancy: {((py_count/img.size)-1)*100:.1f}%)")
            fig.text(0.7, 0.2, info_text, fontsize=14, bbox=dict(facecolor='orange', alpha=0.2))
            
            fig.suptitle(f"Coding Comparison | {Path(f).name}", fontsize=16)
            plt.draw()
            current_idx[0] += 1
        except Exception as e:
            print(f"Error: {e}")
            current_idx[0] += 1
            show_next()

    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close() if e.key=='escape' else show_next())
    show_next()
    plt.show()

if __name__ == '__main__':
    display_coding_comparison('CBIS-DDSM/noise.csv')