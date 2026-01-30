import pydicom
import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from skimage.restoration import estimate_sigma

# --- 1. 定义三种阈值函数 ---
def threshold_hard(subband, lam):
    return pywt.threshold(subband, lam, mode='hard')

def threshold_soft(subband, lam):
    return pywt.threshold(subband, lam, mode='soft')

def threshold_garrote(subband, lam):
    """Garrote 阈值实现: x - lam^2/x if |x| > lam else 0"""
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where(np.abs(subband) > lam, subband - (lam**2)/subband, 0)
    return np.nan_to_num(res)

# --- 2. 核心处理逻辑 ---
def denoise_comparison(img, wavelet='db4', level=3):
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    sigma = estimate_sigma(img)
    # 使用通用阈值 lambda = sigma * sqrt(2*log(N))
    lam = sigma * np.sqrt(2 * np.log(img.size))
    
    results = {}
    methods = {
        'Hard': threshold_hard,
        'Soft': threshold_soft,
        'Garrote': threshold_garrote
    }
    
    for name, func in methods.items():
        new_coeffs = [coeffs[0]] # 保留低频
        for subband_tuple in coeffs[1:]:
            new_coeffs.append(tuple(func(s, lam) for s in subband_tuple))
        results[name] = pywt.waverec2(new_coeffs, wavelet)
    
    return results

# --- 3. 交互显示系统 ---
def display_threshold_comparison(csv_path, num_files=127):
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
            
            # 运行去噪
            denoised = denoise_comparison(img)
            
            imgs = [img, denoised['Hard'], denoised['Soft'], denoised['Garrote']]
            titles = ['Original', 'Hard (Artifacts)', 'Soft (Blurry)', 'Garrote (Sharp/Clean)']

            for i, ax in enumerate(axes):
                ax.clear()
                # 裁剪以观察细节
                h, w = img.shape
                crop = imgs[i][h//3:h//2, w//3:w//2]
                ax.imshow(crop, cmap='bone')
                ax.set_title(titles[i], fontsize=14)
                ax.axis('off')

            fig.suptitle(f"Threshold Method Comparison | Rank: {current_idx[0]+1} | {Path(f).name}", fontsize=15)
            plt.draw()
            current_idx[0] += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
            current_idx[0] += 1
            show_next()

    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close() if e.key=='escape' else show_next())
    show_next()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    display_threshold_comparison('CBIS-DDSM/noise.csv')