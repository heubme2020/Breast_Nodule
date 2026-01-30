import pydicom
import pywt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from pathlib import Path  # 必须导入这个！
from scipy.ndimage import correlate


# 1. 模拟/读取 DICOM (这里使用归一化数据)
def get_dicom_data(file_path):
    ds = pydicom.dcmread(file_path)
    img = ds.pixel_array.astype(np.float32)
    return cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)

# 2. 绘制 Meyer 小波基函数
def plot_meyer_wavelet():
    wavelet = pywt.Wavelet('dmey')

    # 比较不同 level 下 x 的变化
    for lev in [3, 5, 7]:
        phi, psi, x = wavelet.wavefun(level=lev)
        print(f"Level {lev}: x 的点数 = {len(x)}, 范围从 {x[0]:.2f} 到 {x[-1]:.2f}")

    phi, psi, x = wavelet.wavefun(level=7)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, phi)
    plt.title("Scaling Function (Low-pass)") # 负责轮廓
    plt.subplot(1, 2, 2)
    plt.plot(x, psi)
    plt.title("Wavelet Function (High-pass)") # 负责细
    # plt.plot(x, psi, label='Meyer Wavelet (psi)', color='blue')
    # plt.title("Discrete Meyer Wavelet Function")
    plt.grid(True)
    plt.legend()
    plt.show()


def process_and_show_dmey(img, levels=4, ax_list=None):
    """
    修改后的函数：支持在外部提供的 axes 上绘图
    """
    current_img = img
    # 第一张画原图
    ax_list[0].imshow(img, cmap='gray')
    ax_list[0].set_title("Original")
    ax_list[0].axis('off')
    
    # 逐层分解并显示
    for i in range(1, levels + 1):
        # 离散 Meyer 分解 1 层
        c = pywt.wavedec2(current_img, 'dmey', level=1)
        current_img = c[0] 
        
        ax_list[i].imshow(current_img, cmap='bone')
        ax_list[i].set_title(f"L{i} Approx")
        ax_list[i].axis('off')

def display_top_noisy_dicoms(csv_path, num_files=127):
    # 1. 数据准备
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by='NoiseScore', ascending=False)
    top_noisy_files = df_sorted['FullPath'].head(num_files).tolist()
    scores = df_sorted['NoiseScore'].head(num_files).tolist()

    levels = 4 # 建议 4 层，7 层图片太小了
    fig, axes = plt.subplots(1, levels + 1, figsize=(20, 5))
    current_idx = [0]

    def on_key(event):
        """键盘点击事件处理"""
        if event.key == 'escape': # 按 ESC 退出
            plt.close()
            return
        show_next()

    def show_next():
        if current_idx[0] >= len(top_noisy_files):
            print("全部显示完毕。")
            plt.close()
            return

        file_path = top_noisy_files[current_idx[0]]
        score = scores[current_idx[0]]
        
        try:
            # 清除旧图
            for ax in axes:
                ax.clear()

            # 读取 DICOM
            ds = pydicom.dcmread(file_path)
            img = ds.pixel_array.astype(np.float32)
            # 归一化
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

            # 调用小波处理并绘图
            process_and_show_dmey(img, levels=levels, ax_list=axes)
            
            fig.suptitle(f"Rank: {current_idx[0] + 1} | Score: {score:.4f} | File: {Path(file_path).name}")
            plt.draw() # 强制刷新画布
            
            current_idx[0] += 1
        except Exception as e:
            print(f"读取失败 {file_path}: {e}")
            current_idx[0] += 1
            show_next()

    # 注册事件
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # 显示第一张
    show_next()
    
    print("操作说明：窗口激活时，按【任意键】切换下一张，按【ESC】退出。")
    plt.show() # 这里启动主循环


# --- 1. 生成 Morlet 和 Mexican Hat 的核 ---
def get_wavelet_kernels(size=31, sigma=3.0):
    """生成用于图像卷积的小波核"""
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    xx, yy = np.meshgrid(x, y)
    r2 = xx**2 + yy**2
    
    # Mexican Hat (高斯二阶导)
    mexh = (1 - r2/(sigma**2)) * np.exp(-r2/(2*sigma**2))
    
    # Morlet (实部：余弦 * 高斯)
    morl = np.cos(5 * xx / sigma) * np.exp(-r2/(2*sigma**2))
    
    return mexh, morl

# --- 2. 绘制基函数波形 ---
def plot_base_functions():
    wavelet_mexh = pywt.ContinuousWavelet('mexh')
    wavelet_morl = pywt.ContinuousWavelet('morl')
    
    # 获取函数数据
    psi_mexh, x_mexh = wavelet_mexh.wavefun(level=7)
    psi_morl, x_morl = wavelet_morl.wavefun(level=7)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_mexh, psi_mexh, color='red', lw=2)
    plt.title("Mexican Hat Wavelet (Blob Detector)")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_morl, psi_morl, color='blue', lw=2)
    plt.title("Morlet Wavelet (Oscillation/Texture Detector)")
    plt.grid(True)
    plt.show()

# --- 3. 图像处理函数 ---
def process_and_show_cwt_like(img, ax_list):
    mexh_kernel, morl_kernel = get_wavelet_kernels(size=31, sigma=2.0)
    
    # 原始图像
    ax_list[0].imshow(img, cmap='gray')
    ax_list[0].set_title("Original DICOM")
    ax_list[0].axis('off')
    
    # Mexican Hat 卷积结果 - 擅长检测亮点和边缘
    img_mexh = correlate(img, mexh_kernel)
    ax_list[1].imshow(img_mexh, cmap='RdBu_r') # 红蓝配色更易观察正负响应
    ax_list[1].set_title("Mexican Hat Filter\n(Spot/Edge Detection)")
    ax_list[1].axis('off')
    
    # Morlet 卷积结果 - 擅长检测纹理
    img_morl = correlate(img, morl_kernel)
    ax_list[2].imshow(img_morl, cmap='magma')
    ax_list[2].set_title("Morlet Filter\n(Texture/Pattern)")
    ax_list[2].axis('off')

# --- 4. 交互式主程序 ---
def display_special_wavelets(csv_path, num_files=127):
    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    top_files = df['FullPath'].head(num_files).tolist()
    scores = df['NoiseScore'].head(num_files).tolist()

    # 我们显示三列：原图、墨西哥帽、Morlet
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    current_idx = [0]

    def show_next():
        if current_idx[0] >= len(top_files):
            plt.close()
            return

        file_path = top_files[current_idx[0]]
        try:
            for ax in axes: ax.clear()
            
            ds = pydicom.dcmread(file_path)
            img = ds.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
            process_and_show_cwt_like(img, axes)
            
            fig.suptitle(f"Rank: {current_idx[0]+1} | File: {Path(file_path).name}\nKey: Any Key for Next, ESC to Exit", fontsize=14)
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


def process_haar_details(img):
    """
    使用 Haar 小波进行 1 层分解，并提取四个子带特征
    LL: 近似 (低通), LH: 水平细节, HL: 垂直细节, HH: 对角线细节
    """
    # Haar 分解 (level=1)
    coeffs = pywt.wavedec2(img, 'haar', level=1)
    LL, (LH, HL, HH) = coeffs
    
    # 为了显示清晰，对高频分量进行绝对值处理并增强对比度
    def scale_detail(d):
        return np.abs(d) / (np.max(np.abs(d)) + 1e-9)

    return {
        "LL (Approx)": LL,
        "LH (Horizontal)": scale_detail(LH),
        "HL (Vertical)": scale_detail(HL),
        "HH (Diagonal)": scale_detail(HH)
    }

def display_haar_dicoms(csv_path, num_files=127):
    # 1. 加载数据
    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    top_files = df['FullPath'].head(num_files).tolist()
    scores = df['NoiseScore'].head(num_files).tolist()

    # 创建 1x5 布局：原图 + 4个 Haar 子带
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    current_idx = [0]

    def show_next():
        if current_idx[0] >= len(top_files):
            plt.close()
            return

        file_path = top_files[current_idx[0]]
        try:
            for ax in axes: ax.clear()
            
            ds = pydicom.dcmread(file_path)
            img = ds.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
            # 计算 Haar 子带
            haar_results = process_haar_details(img)
            
            # 显示原图
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title("Original DICOM")
            
            # 显示 Haar 子带
            titles = list(haar_results.keys())
            imgs = list(haar_results.values())
            for i in range(4):
                # Haar 分解后尺寸会减半，imshow 会自动拉伸显示
                axes[i+1].imshow(imgs[i], cmap='magma' if i > 0 else 'gray')
                axes[i+1].set_title(titles[i])
            
            for ax in axes: ax.axis('off')
            
            fig.suptitle(f"Haar Decomposition | Rank: {current_idx[0]+1} | File: {Path(file_path).name}", fontsize=14)
            plt.draw()
            current_idx[0] += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            current_idx[0] += 1
            show_next()

    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close() if e.key=='escape' else show_next())
    show_next()
    plt.tight_layout()
    plt.show()

# --- 1. 绘制 db 小波波形 ---
def plot_db_wavelets():
    db_list = ['db2', 'db4', 'db8']
    plt.figure(figsize=(15, 5))
    for i, name in enumerate(db_list):
        wavelet = pywt.Wavelet(name)
        phi, psi, x = wavelet.wavefun(level=7)
        plt.subplot(1, 3, i+1)
        plt.plot(x, psi, color='green')
        plt.title(f"{name} Wavelet Function")
        plt.grid(True)
    plt.show()

# --- 2. 交互式对比预览 ---
def display_db_comparison(csv_path, num_files=127):
    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    files = df['FullPath'].head(num_files).tolist()
    
    # 布局：原图, db2分解, db8分解, 两者差异
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    current_idx = [0]

    def show_next():
        if current_idx[0] >= len(files): return
        f = files[current_idx[0]]
        try:
            for ax in axes: ax.clear()
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
            # 使用 db2 处理 (低阶，边缘较硬)
            c2 = pywt.wavedec2(img, 'db2', level=2)
            recon_db2 = pywt.waverec2(c2, 'db2')
            
            # 使用 db8 处理 (高阶，边缘较平滑)
            c8 = pywt.wavedec2(img, 'db8', level=2)
            recon_db8 = pywt.waverec2(c8, 'db8')
            
            # 计算两者残差（放大差异）
            diff = np.abs(recon_db2 - recon_db8)
            diff = (diff - np.min(diff)) / (np.max(diff) + 1e-9)

            imgs = [img, recon_db2, recon_db8, diff]
            titles = ['Original', 'db2 (Low Order)', 'db8 (High Order)', 'Difference (db2 vs db8)']

            for i, ax in enumerate(axes):
                h, w = img.shape
                crop = imgs[i][h//3:h//2, w//3:w//2]
                ax.imshow(crop, cmap='bone' if i < 3 else 'hot')
                ax.set_title(titles[i])
                ax.axis('off')

            fig.suptitle(f"Daubechies Comparison | {Path(f).name}", fontsize=15)
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


# --- 1. 绘制波形对比 ---
def plot_wavelet_shapes():
    # sym4 的支撑长度为 7，coif2 的支撑长度为 11
    wavelets = [pywt.Wavelet('sym4'), pywt.Wavelet('coif2')]
    titles = ['Symlet 4 (Near Symmetric)', 'Coiflet 2 (Double Vanishing)']
    
    plt.figure(figsize=(12, 5))
    for i, wav in enumerate(wavelets):
        phi, psi, x = wav.wavefun(level=7)
        plt.subplot(1, 2, i+1)
        plt.plot(x, psi, color='teal' if i==0 else 'chocolate', lw=2)
        plt.title(titles[i])
        plt.grid(True)
    plt.show()

# --- 2. 图像处理对比逻辑 ---
def denoise_with_wavelet(img, wav_name, levels=3):
    coeffs = pywt.wavedec2(img, wav_name, level=levels)
    # 使用通用硬阈值去噪，以便观察相位对边缘的影响
    sigma = 0.02 # 设定一个统一的阈值强度
    new_coeffs = [coeffs[0]]
    for level in coeffs[1:]:
        new_coeffs.append(tuple(pywt.threshold(c, sigma, mode='hard') for c in level))
    return pywt.waverec2(new_coeffs, wav_name)

# --- 3. 交互式显示系统 ---
def display_sym_coif_comparison(csv_path, num_files=127):
    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    files = df['FullPath'].head(num_files).tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    current_idx = [0]

    def show_next():
        if current_idx[0] >= len(files): return
        f = files[current_idx[0]]
        try:
            for ax in axes: ax.clear()
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
            res_sym = denoise_with_wavelet(img, 'sym4')
            res_coif = denoise_with_wavelet(img, 'coif2')
            
            imgs = [img, res_sym, res_coif]
            titles = ['Original', 'Symlets (sym4)\nReduced Phase Shift', 'Coiflets (coif2)\nHigh Approximation']
            
            for i in range(3):
                h, w = img.shape
                # 重点看边缘细节
                crop = imgs[i][h//3:h//2, w//3:w//2]
                axes[i].imshow(crop, cmap='bone')
                axes[i].set_title(titles[i])
                axes[i].axis('off')
                
            fig.suptitle(f"Symlets vs Coiflets | {Path(f).name}", fontsize=15)
            plt.draw()
            current_idx[0] += 1
        except Exception as e:
            print(f"Error: {e}"); current_idx[0] += 1; show_next()

    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close() if e.key=='escape' else show_next())
    show_next()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # display_sym_coif_comparison('CBIS-DDSM/noise.csv')
    # plot_wavelet_shapes() 
    # display_db_comparison('CBIS-DDSM/noise.csv')
    # plot_db_wavelets()
    # display_haar_dicoms('CBIS-DDSM/noise.csv')
    # display_special_wavelets('CBIS-DDSM/noise.csv')
    # plot_base_functions()
    # display_top_noisy_dicoms('CBIS-DDSM/noise.csv')
    plot_meyer_wavelet()

