import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.signal import wiener
from skimage.restoration import estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- 1. 维纳滤波核心逻辑 ---
def apply_wiener_denoising(img, mysize=7):
    """
    执行两种维纳滤波：
    1. 自动噪声估计 (SciPy 默认)
    2. 增强型 (使用特定噪声水平)
    """
    # 转换为 float64 以防溢出
    img_float = img.astype(np.float64)
    
    # 自动估计噪声标准差 (使用之前学到的中位数绝对偏差法)
    sigma_est = estimate_sigma(img_float)
    noise_var = sigma_est ** 2
    
    # 方法 A: 纯自动维纳滤波 (窗口设为 mysize)
    res_auto = wiener(img_float, mysize=mysize)
    
    # 方法 B: 引导噪声方差维纳滤波 (手动传入估计的 noise_var)
    # 虽然 SciPy 的 wiener 函数内部有 noise 参数，但其表现与实现略有不同
    res_guided = wiener(img_float, mysize=mysize, noise=noise_var)
    
    return {
        'Auto': np.clip(res_auto, 0, 1),
        'Guided': np.clip(res_guided, 0, 1),
        'Sigma': sigma_est
    }

# --- 2. 交互显示系统 ---
def display_wiener_comparison(csv_path, num_files=127):
    if not Path(csv_path).exists():
        print(f"找不到 CSV 文件: {csv_path}")
        return

    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    files = df['FullPath'].head(num_files).tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    current_idx = [0]

    def show_next():
        if current_idx[0] >= len(files): 
            print("所有图像预览结束。")
            plt.close()
            return
            
        f = files[current_idx[0]]
        try:
            # 读取 DICOM
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            # 基础归一化到 [0, 1]
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-9)
            
            # 执行维纳滤波处理 (窗口取 5x5 以平衡平滑度和细节)
            results = apply_wiener_denoising(img, mysize=5)
            
            # 计算 PSNR (以原始图像为基准看噪声压制程度)
            psnr_auto = psnr(img, results['Auto'], data_range=1.0)
            psnr_guided = psnr(img, results['Guided'], data_range=1.0)

            imgs = [img, results['Auto'], results['Guided']]
            titles = [
                'Original (Noisy)', 
                f'Wiener Auto\nPSNR: {psnr_auto:.2f}dB', 
                f'Wiener Guided\nPSNR: {psnr_guided:.2f}dB'
            ]

            # 渲染画面
            for i, ax in enumerate(axes):
                ax.clear()
                # 裁剪中心区域以观察微小钙化点或组织纹理
                h, w = img.shape
                crop = imgs[i][h//3:int(h//1.5), w//3:int(w//1.5)]
                ax.imshow(crop, cmap='bone')
                ax.set_title(titles[i], fontsize=12)
                ax.axis('off')

            fig.suptitle(f"Wiener Filtering | Est. Sigma: {results['Sigma']:.4f}\n{Path(f).name}", fontsize=14)
            plt.draw()
            current_idx[0] += 1
            
        except Exception as e:
            print(f"处理错误 {f}: {e}")
            current_idx[0] += 1
            show_next()

    # 绑定按键
    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close() if e.key=='escape' else show_next())
    show_next()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 修改为你实际的 CSV 路径
    display_wiener_comparison('CBIS-DDSM/noise.csv')