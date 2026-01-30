import pandas as pd
import pydicom
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
from pathlib import Path

# --- 算法部分：TIW ---
def denoise_tiw(img, wavelet='dmey', levels=3, max_shift=2): # 减到位移2以提高预览速度
    avg_img = np.zeros_like(img)
    count = 0
    for s_x in range(max_shift):
        for s_y in range(max_shift):
            shifted_img = np.roll(np.roll(img, s_x, axis=0), s_y, axis=1)
            coeffs = pywt.wavedec2(shifted_img, wavelet, level=levels)
            sigma = estimate_sigma(shifted_img)
            threshold = sigma * np.sqrt(2 * np.log(shifted_img.size))
            
            new_coeffs = [coeffs[0]] + [tuple(pywt.threshold(c, threshold, mode='soft') for c in level) 
                                       for level in coeffs[1:]]
            denoised = pywt.waverec2(new_coeffs, wavelet)
            avg_img += np.roll(np.roll(denoised, -s_x, axis=0), -s_y, axis=1)
            count += 1
    return avg_img / count

# --- 算法部分：FWB ---
def denoise_fwb_simplified(img, wavelet='dmey', levels=3):
    coeffs = pywt.wavedec2(img, wavelet, level=levels)
    sigma_n = estimate_sigma(img)
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        level_coeffs = []
        for subband in coeffs[i]:
            var_y = np.mean(subband**2)
            sigma_s = np.sqrt(max(var_y - sigma_n**2, 0))
            threshold = (sigma_n**2) / sigma_s if sigma_s > 0 else np.max(np.abs(subband))
            level_coeffs.append(pywt.threshold(subband, threshold, mode='soft'))
        new_coeffs.append(tuple(level_coeffs))
    return pywt.waverec2(new_coeffs, wavelet)

# --- 主逻辑：读取CSV并交互显示 ---
def display_denoising_comparison(csv_path, num_files=127):
    # 1. 加载并排序
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by='NoiseScore', ascending=False)
    top_files = df_sorted['FullPath'].head(num_files).tolist()
    scores = df_sorted['NoiseScore'].head(num_files).tolist()

    # 2. 初始化画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    current_idx = [0]

    def show_next():
        if current_idx[0] >= len(top_files):
            print("已经看完所有文件。")
            plt.close()
            return

        file_path = top_files[current_idx[0]]
        score = scores[current_idx[0]]
        
        try:
            # 读取并预处理 DICOM
            ds = pydicom.dcmread(file_path)
            img = ds.pixel_array.astype(np.float32)
            # 基础归一化
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
            # 运行去噪算法
            tiw_img = denoise_tiw(img)
            fwb_img = denoise_fwb_simplified(img)

            # 绘制图像
            imgs = [img, tiw_img, fwb_img]
            titles = ['Original (Noisy)', 'TIW (Cycle Spinning)', 'FWB (BayesShrink)']
            
            for i in range(3):
                axes[i].clear()
                # 显示局部放大，更容易看出算法差异（可选：去掉切片显示全图）
                h, w = img.shape
                crop = imgs[i][h//4:h//2, w//4:w//2] 
                axes[i].imshow(crop, cmap='bone')
                axes[i].set_title(titles[i])
                axes[i].axis('off')

            fig.suptitle(f"Rank: {current_idx[0]+1}/{num_files} | Noise: {score:.4f}\nFile: {Path(file_path).name}")
            plt.draw()
            current_idx[0] += 1
            
        except Exception as e:
            print(f"处理失败 {file_path}: {e}")
            current_idx[0] += 1
            show_next()

    def on_key(event):
        if event.key == 'escape':
            plt.close()
        else:
            show_next()

    # 绑定按键事件
    fig.canvas.mpl_connect('key_press_event', on_key)
    show_next() # 显示第一张
    
    print(">>> 已启动交互窗口。点击图像后按【任意键】切换，按【ESC】退出。")
    plt.show()

if __name__ == '__main__':
    CSV_FILE = 'CBIS-DDSM/noise.csv' # 确保此路径正确
    display_denoising_comparison(CSV_FILE)