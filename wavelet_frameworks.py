import pydicom
import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def compare_wavelet_frameworks(img):
    level = 2
    
    # --- 1. 正交基 (Orthogonal) - 使用 db4 ---
    # 特点：紧凑，重构时矩阵转置即可
    coeffs_ortho = pywt.wavedec2(img, 'db4', level=level)
    recon_ortho = pywt.waverec2(coeffs_ortho, 'db4')

    # --- 2. 双正交基 (Biorthogonal) - 使用 bior2.2 ---
    # 特点：分解和重构滤镜对称，解决相位失真
    coeffs_bior = pywt.wavedec2(img, 'bior2.2', level=level)
    recon_bior = pywt.waverec2(coeffs_bior, 'bior2.2')

    # --- 3. 框架/过完备 (Frames) - 使用 SWT (非下采样小波) ---
    # 特点：没有下采样，系数极多，平移不变
    # 注意：SWT 要求图像尺寸必须是 2^level 的倍数
    h, w = img.shape
    new_h = (h // 2**level) * 2**level
    new_w = (w // 2**level) * 2**level
    img_pad = img[:new_h, :new_w]
    
    coeffs_frame = pywt.swt2(img_pad, 'db4', level=level)
    recon_frame = pywt.iswt2(coeffs_frame, 'db4')

    return (img_pad, recon_ortho[:new_h, :new_w], 
            recon_bior[:new_h, :new_w], recon_frame)

# --- 3. 交互式显示系统 ---
def display_framework_comparison(csv_path, num_files=127):
    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    files = df['FullPath'].head(num_files).tolist()
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    current_idx = [0]

    def show_next():
        if current_idx[0] >= len(files): return
        f = files[current_idx[0]]
        try:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
            orig, ortho, bior, frame = compare_wavelet_frameworks(img)
            
            # 为了对比去噪能力，我们在处理前加一点噪声（模拟真实去噪应用）
            titles = ['Original', 'Orthogonal (db4)', 'Biorthogonal (bior2.2)', 'Frames (SWT)']
            imgs = [orig, ortho, bior, frame]

            for i, ax in enumerate(axes):
                ax.clear()
                # 重点看边缘细节
                h, w = orig.shape
                crop = imgs[i][h//3:h//2, w//3:w//2]
                ax.imshow(crop, cmap='bone')
                ax.set_title(titles[i])
                ax.axis('off')

            fig.suptitle(f"Framework Comparison | File: {Path(f).name}", fontsize=15)
            plt.draw()
            current_idx[0] += 1
        except Exception as e:
            print(f"Error: {e}")
            current_idx[0] += 1
            show_next()

    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close() if e.key=='escape' else show_next())
    show_next()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    display_framework_comparison('CBIS-DDSM/noise.csv')