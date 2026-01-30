import pandas as pd
import pydicom
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
from pathlib import Path
from scipy.signal import convolve2d

# --- 1. Visu Shrink (Universal) ---
def visu_shrink(subband, sigma, n):
    threshold = sigma * np.sqrt(2 * np.log(n))
    return pywt.threshold(subband, threshold, mode='soft')

# --- 2. Sure Shrink (Adaptive) ---
def sure_shrink(subband, sigma):
    n = subband.size
    sx2 = np.sort(np.abs(subband).flatten())**2
    risk = (n - 2 * np.arange(n) + (np.cumsum(sx2) + np.arange(n, 0, -1) * sx2)) / n
    best_idx = np.argmin(risk)
    threshold = np.sqrt(sx2[best_idx])
    return pywt.threshold(subband, threshold, mode='soft')

# --- 3. Bayes Shrink (Classical) ---
def bayes_shrink(subband, sigma_n):
    var_y = np.mean(subband**2)
    sigma_s = np.sqrt(max(var_y - sigma_n**2, 0))
    threshold = (sigma_n**2) / sigma_s if sigma_s > 0 else np.max(np.abs(subband))
    return pywt.threshold(subband, threshold, mode='soft')

# --- 4. Prob Shrink (Probability based) ---
def prob_shrink(subband, sigma_n):
    # 简单的概率估计：基于局部方差与噪声方差的比值
    var_y = subband**2
    # 假设信号服从拉普拉斯分布后的收缩因子
    shrink_factor = np.maximum(0, 1 - (sigma_n**2 / (var_y + 1e-9)))
    return subband * shrink_factor

# --- 5. Neigh Shrink (3x3 Neighborhood) ---
def neigh_shrink(subband, sigma_n):
    n = subband.size
    threshold2 = 2 * (sigma_n**2) * np.log(n)
    # 计算 3x3 邻域内的平方和
    from scipy.signal import convolve2d
    kernel = np.ones((3, 3))
    s2 = convolve2d(subband**2, kernel, mode='same')
    # 收缩因子: (1 - lambda^2 / S^2)_+
    with np.errstate(divide='ignore', invalid='ignore'):
        shrink = np.maximum(0, 1 - (threshold2 / (s2 + 1e-9)))
    return subband * shrink

# --- 6. Block Shrink (Block based) ---
def block_shrink(subband, sigma_n, block_size=4):
    h, w = subband.shape
    output = np.zeros_like(subband)
    # L = sigma^2 * log(N)
    threshold2 = 4.5 * (sigma_n**2) # 经验参数
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = subband[i:i+block_size, j:j+block_size]
            s2 = np.sum(block**2)
            shrink = max(0, 1 - (threshold2 / (s2 + 1e-9)))
            output[i:i+block_size, j:j+block_size] = block * shrink
    return output

# --- 算法 1: VisuShrink (Universal Threshold) ---
def denoise_visu_shrink(img, wavelet='dmey', levels=3):
    coeffs = pywt.wavedec2(img, wavelet, level=levels)
    # 估计全局噪声标准差 sigma
    sigma = estimate_sigma(img)
    # 计算 Universal Threshold: lambda = sigma * sqrt(2 * log(N))
    n_pixels = img.size
    threshold = sigma * np.sqrt(2 * np.log(n_pixels))
    
    # 对所有高频子带应用相同的阈值
    new_coeffs = [coeffs[0]] # 保留 LL
    for level in coeffs[1:]:
        new_coeffs.append(tuple(pywt.threshold(c, threshold, mode='soft') for c in level))
    
    return pywt.waverec2(new_coeffs, wavelet)

# --- 算法 2: BayesShrink (用于对比) ---
def denoise_bayes_shrink(img, wavelet='dmey', levels=3):
    coeffs = pywt.wavedec2(img, wavelet, level=levels)
    sigma_n = estimate_sigma(img)
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        level_coeffs = []
        for subband in coeffs[i]:
            var_y = np.mean(subband**2)
            sigma_s = np.sqrt(max(var_y - sigma_n**2, 0))
            # 贝叶斯阈值是自适应的
            threshold = (sigma_n**2) / sigma_s if sigma_s > 0 else np.max(np.abs(subband))
            level_coeffs.append(pywt.threshold(subband, threshold, mode='soft'))
        new_coeffs.append(tuple(level_coeffs))
    return pywt.waverec2(new_coeffs, wavelet)

# --- 交互显示逻辑 ---
def display_visu_comparison(csv_path, num_files=127):
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by='NoiseScore', ascending=False)
    top_files = df_sorted['FullPath'].head(num_files).tolist()
    scores = df_sorted['NoiseScore'].head(num_files).tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    current_idx = [0]

    def show_next():
        if current_idx[0] >= len(top_files):
            plt.close()
            return

        file_path = top_files[current_idx[0]]
        try:
            ds = pydicom.dcmread(file_path)
            img = ds.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
            # 处理
            visu_res = denoise_visu_shrink(img)
            bayes_res = denoise_bayes_shrink(img)

            imgs = [img, visu_res, bayes_res]
            titles = ['Noisy DICOM', 'VisuShrink (Universal)', 'BayesShrink (Adaptive)']
            
            for i in range(3):
                axes[i].clear()
                # 裁剪中心区域观察边缘丢失情况
                h, w = img.shape
                crop = imgs[i][h//3:h//2, w//3:w//2]
                axes[i].imshow(crop, cmap='bone')
                axes[i].set_title(titles[i])
                axes[i].axis('off')

            fig.suptitle(f"Rank: {current_idx[0]+1} | File: {Path(file_path).name}\nNote: Observe how VisuShrink smooths out details.")
            plt.draw()
            current_idx[0] += 1
        except Exception as e:
            print(f"Error: {e}")
            current_idx[0] += 1
            show_next()

    fig.canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else show_next())
    show_next()
    plt.show()

def run_all_shrinks(img, levels=3):
    wavelet = 'dmey'
    coeffs = pywt.wavedec2(img, wavelet, level=levels)
    sigma_n = estimate_sigma(img)
    n = img.size
    
    results = {}
    methods = {
        'Visu': lambda s: visu_shrink(s, sigma_n, n),
        'Sure': lambda s: sure_shrink(s, sigma_n),
        'Bayes': lambda s: bayes_shrink(s, sigma_n),
        'Prob': lambda s: prob_shrink(s, sigma_n),
        'Neigh': lambda s: neigh_shrink(s, sigma_n),
        'Block': lambda s: block_shrink(s, sigma_n)
    }

    for name, func in methods.items():
        new_coeffs = [coeffs[0]]
        for level in coeffs[1:]:
            new_coeffs.append(tuple(func(sub) for sub in level))
        results[name] = pywt.waverec2(new_coeffs, wavelet)
    
    return results

def display_full_comparison(csv_path, num_files=127):
    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    files = df['FullPath'].head(num_files).tolist()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    current_idx = [0]

    def update():
        if current_idx[0] >= len(files): return
        f = files[current_idx[0]]
        ds = pydicom.dcmread(f)
        img = ds.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        # 批量去噪
        denoised_data = run_all_shrinks(img)
        
        all_imgs = [img] + list(denoised_data.values())
        all_titles = ['Original'] + list(denoised_data.keys())

        for i, ax in enumerate(axes):
            ax.clear()
            if i < len(all_imgs):
                # 局部放大显示细节
                h, w = img.shape
                crop = all_imgs[i][h//3:h//2, w//3:w//2]
                ax.imshow(crop, cmap='bone')
                ax.set_title(all_titles[i], fontsize=14)
            ax.axis('off')
        
        fig.suptitle(f"Rank {current_idx[0]+1}: {Path(f).name}", fontsize=16)
        plt.draw()
        current_idx[0] += 1

    fig.canvas.mpl_connect('key_press_event', lambda e: update())
    update()
    plt.tight_layout()
    plt.show()


def process_all_strategies(img, wavelet='db4', levels=3):
    coeffs = pywt.wavedec2(img, wavelet, level=levels)
    sigma_n = estimate_sigma(img)
    n = img.size
    
    methods = {
        'Visu': lambda s: visu_shrink(s, sigma_n, n),
        'SURE': lambda s: sure_shrink(s, sigma_n),
        'Bayes': lambda s: bayes_shrink(s, sigma_n),
        'Neigh/CV': lambda s: neigh_shrink(s, sigma_n)
    }
    
    results = {}
    for name, func in methods.items():
        new_coeffs = [coeffs[0]]
        for level_tuple in coeffs[1:]:
            new_coeffs.append(tuple(func(s) for s in level_tuple))
        results[name] = pywt.waverec2(new_coeffs, wavelet)
    return results

def display_strategy_comparison(csv_path, num_files=127):
    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    files = df['FullPath'].head(num_files).tolist()
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    current_idx = [0]

    def update():
        if current_idx[0] >= len(files): return
        f = files[current_idx[0]]
        try:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
            denoised = process_all_strategies(img)
            
            all_imgs = [img] + list(denoised.values())
            all_titles = ['Original'] + list(denoised.keys())

            for i, ax in enumerate(axes):
                ax.clear()
                # 局部放大，重点看肿块边缘和背景噪点的剥离
                h, w = img.shape
                crop = all_imgs[i][h//3:h//2, w//3:w//2]
                ax.imshow(crop, cmap='bone')
                ax.set_title(all_titles[i], fontsize=14)
                ax.axis('off')
            
            fig.suptitle(f"Threshold Strategy Comparison | File: {Path(f).name}", fontsize=16)
            plt.draw()
            current_idx[0] += 1
        except Exception as e:
            print(f"Skip {f}: {e}")
            current_idx[0] += 1
            update()

    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close() if e.key=='escape' else update())
    update()
    plt.tight_layout()
    plt.show()

# --- 1. 传统 VisuShrink ---
def visu_shrink_logic(subband, sigma, n):
    lam = sigma * np.sqrt(2 * np.log(n))
    return pywt.threshold(subband, lam, mode='soft')

# --- 2. Bivariate Shrink (核心实现) ---
def bivariate_shrink_logic(y, y_parent, sigma_n):
    """
    y: 当前层子带 (儿子)
    y_parent: 上一层子带 (父亲)，需通过插值对齐尺寸
    sigma_n: 噪声标准差
    """
    # 估计信号的标准差 sigma_s (基于局部窗口或经验公式)
    # 这里使用简化的 MAP 估计
    var_y = np.mean(y**2)
    sigma_s = np.sqrt(max(var_y - sigma_n**2, 1e-9))
    
    lam = np.sqrt(3) * (sigma_n**2) / sigma_s
    
    # 计算能量模长
    magnitude = np.sqrt(y**2 + y_parent**2)
    
    # 应用收缩公式
    shrink_factor = np.maximum(0, 1 - lam / (magnitude + 1e-9))
    return y * shrink_factor

# --- 3. 图像处理调度器 ---
def process_comparison(img, levels=3):
    wavelet = 'db4'
    coeffs = pywt.wavedec2(img, wavelet, level=levels)
    sigma_n = estimate_sigma(img)
    n = img.size
    
    # 结果 1: VisuShrink
    res_visu_coeffs = [coeffs[0]]
    for level_tuple in coeffs[1:]:
        res_visu_coeffs.append(tuple(visu_shrink_logic(s, sigma_n, n) for s in level_tuple))
    res_visu = pywt.waverec2(res_visu_coeffs, wavelet)
    
    # 结果 2: Bivariate Shrink
    res_biv_coeffs = [coeffs[0]]
    # 从高层往低层处理，因为需要父亲信息
    # coeffs 的顺序是 [LL, (LHn, HLn, HHn), ..., (LH1, HL1, HH1)]
    for i in range(1, len(coeffs)):
        current_level = coeffs[i]
        if i == 1: # 最高层没有父亲，退化为传统处理
            res_biv_coeffs.append(tuple(visu_shrink_logic(s, sigma_n, n) for s in current_level))
        else:
            parent_level = res_biv_coeffs[i-1] # 取已经处理过的父亲层
            new_level = []
            for s_idx in range(3): # LH, HL, HH
                son = current_level[s_idx]
                parent = parent_level[s_idx]
                # 对父亲进行上采样以匹配儿子尺寸
                parent_up = parent.repeat(2, axis=0).repeat(2, axis=1)
                # 确保尺寸一致 (处理奇数像素边缘)
                parent_up = parent_up[:son.shape[0], :son.shape[1]]
                
                new_level.append(bivariate_shrink_logic(son, parent_up, sigma_n))
            res_biv_coeffs.append(tuple(new_level))
            
    res_biv = pywt.waverec2(res_biv_coeffs, wavelet)
    return res_visu, res_biv

# --- 4. 交互式对比预览 ---
def display_biv_comparison(csv_path, num_files=127):
    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    files = df['FullPath'].head(num_files).tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    current_idx = [0]

    def update():
        if current_idx[0] >= len(files): return
        f = files[current_idx[0]]
        try:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
            visu, biv = process_comparison(img)
            
            imgs = [img, visu, biv]
            titles = ['Original (Noisy)', 'VisuShrink (Isolated)', 'Bivariate (Edge Preserved)']
            
            for i, ax in enumerate(axes):
                ax.clear()
                # 裁剪中心区域看微小特征
                h, w = img.shape
                crop = imgs[i][h//3:h//2, w//3:w//2]
                ax.imshow(crop, cmap='bone')
                ax.set_title(titles[i])
                ax.axis('off')
                
            fig.suptitle(f"Bivariate vs Universal | {Path(f).name}", fontsize=15)
            plt.draw()
            current_idx[0] += 1
        except Exception as e:
            print(f"Error: {e}"); current_idx[0] += 1; update()

    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close() if e.key=='escape' else update())
    update()
    plt.show()

# --- 3. ModiNeighShrink (Improved) ---
def modi_neigh_shrink(subband, sigma_n):
    n = subband.size
    # 改进点：使用更小的经验阈值或自适应调整
    lam_modi2 = (sigma_n**2) * np.log(n) # 减小阈值权重
    kernel = np.ones((3, 3))
    s2 = convolve2d(subband**2, kernel, mode='same')
    # 修正后的收缩逻辑
    shrink = np.maximum(0, 1 - lam_modi2 / (s2 + 1e-9))
    return subband * shrink

# --- 4. 交互式对比系统 ---
def display_neigh_comparison(csv_path, num_files=127):
    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    files = df['FullPath'].head(num_files).tolist()
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    current_idx = [0]

    def update():
        if current_idx[0] >= len(files): return
        f = files[current_idx[0]]
        try:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
            # 分解
            wavelet = 'db4'
            coeffs = pywt.wavedec2(img, wavelet, level=3)
            sigma_n = estimate_sigma(img)
            n_pix = img.size
            
            results = {}
            methods = {
                'Visu': lambda s: visu_shrink(s, sigma_n, n_pix),
                'Neigh': lambda s: neigh_shrink(s, sigma_n),
                'ModiNeigh': lambda s: modi_neigh_shrink(s, sigma_n)
            }
            
            for name, func in methods.items():
                new_c = [coeffs[0]]
                for level in coeffs[1:]:
                    new_c.append(tuple(func(s) for s in level))
                results[name] = pywt.waverec2(new_c, wavelet)

            imgs = [img, results['Visu'], results['Neigh'], results['ModiNeigh']]
            titles = ['Original', 'Visu (Blurry)', 'Neigh (Stable)', 'ModiNeigh (Best Balance)']

            for i, ax in enumerate(axes):
                ax.clear()
                h, w = img.shape
                crop = imgs[i][h//3:h//2, w//3:w//2]
                ax.imshow(crop, cmap='bone')
                ax.set_title(titles[i])
                ax.axis('off')

            fig.suptitle(f"Neighborhood Methods Comparison | {Path(f).name}")
            plt.draw()
            current_idx[0] += 1
        except Exception as e:
            print(f"Error: {e}"); current_idx[0] += 1; update()

    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close() if e.key=='escape' else update())
    update()
    plt.show()

if __name__ == '__main__':
    # display_visu_comparison('CBIS-DDSM/noise.csv')
    # display_full_comparison('CBIS-DDSM/noise.csv')
    # display_strategy_comparison('CBIS-DDSM/noise.csv')
    # display_biv_comparison('CBIS-DDSM/noise.csv')
    display_neigh_comparison('CBIS-DDSM/noise.csv')


