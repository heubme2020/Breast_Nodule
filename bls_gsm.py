# import numpy as np
# import pydicom
# import pywt
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# from scipy.signal import convolve2d
# from skimage.restoration import estimate_sigma

# def bls_gsm_denoise_block(block, sigma_n):
#     """
#     BLS-GSM 的核心简化逻辑：局部维纳滤波的非线性扩展
#     """
#     # 1. 计算局部协方差
#     C_y = np.dot(block, block.T) / block.shape[1]
    
#     # 2. 估计信号协方差 C_x = C_y - C_noise
#     # 假设噪声是独立的，C_noise = sigma_n^2 * I
#     C_n = (sigma_n**2) * np.eye(block.shape[0])
#     C_x = C_y - C_n
    
#     # 特征值分解保证正定性
#     vals, vecs = np.linalg.eigh(C_x)
#     vals = np.maximum(vals, 0)
#     C_x = np.dot(vecs * vals, vecs.T)
    
#     # 3. 贝叶斯估计量 (维纳增益矩阵)
#     # Gain = C_x * (C_x + C_n)^-1
#     try:
#         gain = np.dot(C_x, np.linalg.inv(C_x + C_n))
#         # 估计中心像素
#         center_idx = block.shape[0] // 2
#         return np.dot(gain, block)[center_idx, :]
#     except:
#         return block[block.shape[0]//2, :]

# def denoise_bls_gsm_approx(img, sigma_n, window_size=3):
#     """
#     近似 BLS-GSM 处理：在空间域模拟局部混合高斯建模
#     """
#     h, w = img.shape
#     half_win = window_size // 2
#     img_pad = np.pad(img, half_win, mode='reflect')
#     output = np.zeros_like(img)
    
#     # 展开邻域向量 (以 3x3 为例，每个像素对应一个 9 维向量)
#     for i in range(h):
#         for j in range(w):
#             patch = img_pad[i:i+window_size, j:j+window_size].flatten()
#             # 为了计算协方差，通常需要局部统计，这里演示其核心思想
#             # 实际 BLS-GSM 在小波子带内进行，此处为演示其收缩效果
#             var_y = np.var(patch)
#             gain = max(0, (var_y - sigma_n**2) / (var_y + 1e-9))
#             output[i, j] = img[i, j] * gain
            
#     return output

# # --- 3. 交互对比展示 ---
# def display_bls_gsm_comparison(csv_path, num_files=127):
#     df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
#     files = df['FullPath'].head(num_files).tolist()
    
#     fig, axes = plt.subplots(1, 3, figsize=(18, 7))
#     current_idx = [0]

#     def update():
#         if current_idx[0] >= len(files): return
#         f = files[current_idx[0]]
#         try:
#             ds = pydicom.dcmread(f)
#             img = ds.pixel_array.astype(np.float32)
#             img = (img - np.min(img)) / (np.max(img) - np.min(img))
            
#             sigma_n = estimate_sigma(img)
            
#             # 1. 传统 BayesShrink (小波域对比)
#             coeffs = pywt.wavedec2(img, 'db4', level=3)
#             new_c = [coeffs[0]]
#             for level in coeffs[1:]:
#                 new_c.append(tuple(pywt.threshold(s, sigma_n**2/np.std(s), mode='soft') for s in level))
#             res_bayes = pywt.waverec2(new_c, 'db4')
            
#             # 2. BLS-GSM 效果模拟
#             res_bls = denoise_bls_gsm_approx(img, sigma_n)

#             imgs = [img, res_bayes, res_bls]
#             titles = ['Original Noisy', 'BayesShrink (Wavelet)', 'BLS-GSM (Statistical Modeling)']

#             for i, ax in enumerate(axes):
#                 ax.clear()
#                 h, w = img.shape
#                 crop = imgs[i][h//3:h//2, w//3:w//2]
#                 ax.imshow(crop, cmap='bone')
#                 ax.set_title(titles[i])
#                 ax.axis('off')
                
#             fig.suptitle(f"The Golden Standard: BLS-GSM | {Path(f).name}")
#             plt.draw()
#             current_idx[0] += 1
#         except Exception as e:
#             print(f"Error: {e}"); current_idx[0] += 1; update()

#     fig.canvas.mpl_connect('key_press_event', lambda e: plt.close() if e.key=='escape' else update())
#     update()
#     plt.show()

# if __name__ == '__main__':
#     display_bls_gsm_comparison('CBIS-DDSM/noise.csv')
import numpy as np
import pydicom
import pywt
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter
from skimage.restoration import estimate_sigma

def fast_bls_gsm_approx(img, sigma_n, wavelet='db4', level=3):
    """
    高性能小波域局部统计去噪 (BLS-GSM 的核心向量化实现)
    """
    # 1. 多尺度小波分解
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    new_coeffs = [coeffs[0]]  # 低频部分通常保留

    # 2. 对每个高频子带进行局部自适应收缩
    for i in range(1, len(coeffs)):
        denoised_bands = []
        for band in coeffs[i]:
            # 计算局部能量分布 E[y^2]
            # 使用 3x3 窗口进行快速均值滤波统计
            local_energy = uniform_filter(band**2, size=3)
            
            # 估计信号方差 sigma_x^2 = max(0, E[y^2] - sigma_n^2)
            # 这里的 sigma_n 随尺度微调效果更好，此处使用全局估计值
            sigma_x2 = np.maximum(local_energy - sigma_n**2, 0)
            
            # 计算维纳增益：Gain = sigma_x^2 / (sigma_x^2 + sigma_n^2)
            gain = sigma_x2 / (sigma_x2 + sigma_n**2 + 1e-10)
            
            # 应用增益并保存
            denoised_bands.append(band * gain)
        new_coeffs.append(tuple(denoised_bands))

    # 3. 小波逆变换重构
    return pywt.waverec2(new_coeffs, wavelet)

def display_bls_gsm_comparison(csv_path, num_files=127):
    # 读取数据
    if not Path(csv_path).exists():
        print(f"Error: 找不到文件 {csv_path}")
        return
        
    df = pd.read_csv(csv_path).sort_values(by='NoiseScore', ascending=False)
    files = df['FullPath'].head(num_files).tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    plt.subplots_adjust(wspace=0.1)
    current_idx = [0]

    def update():
        if current_idx[0] >= len(files): 
            print("已到达最后一个文件。")
            return
            
        f = files[current_idx[0]]
        try:
            # 读取 DICOM
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            
            # 归一化到 [0, 1]
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
            
            # 自动估计噪声标准差
            sigma_n = estimate_sigma(img, average_sigmas=True)
            
            # --- 算法对比 ---
            # 1. 传统 BayesShrink (基于小波阈值)
            coeffs_b = pywt.wavedec2(img, 'db4', level=3)
            new_c_b = [coeffs_b[0]]
            for level in coeffs_b[1:]:
                # 典型的 BayesShrink 阈值公式
                new_c_b.append(tuple(pywt.threshold(s, (sigma_n**2)/np.maximum(np.std(s), 1e-9), mode='soft') for s in level))
            res_bayes = pywt.waverec2(new_c_b, 'db4')
            
            # 2. 优化后的 BLS-GSM (局部统计建模)
            res_bls = fast_bls_gsm_approx(img, sigma_n)

            # --- 显示结果 ---
            imgs = [img, res_bayes, res_bls]
            titles = ['Original Noisy', 'Wavelet BayesShrink', 'BLS-GSM (Statistical)']

            for i, ax in enumerate(axes):
                ax.clear()
                # 针对医学影像进行中心切片放大观察细节
                h, w = img.shape
                crop = imgs[i][h//3:h//2, w//3:w//2]
                ax.imshow(crop, cmap='bone')
                ax.set_title(titles[i], fontsize=12)
                ax.axis('off')
                
            fig.suptitle(f"Medical Image Denoising | {Path(f).name}\nPress 'Space' for Next, 'Esc' to Quit", fontsize=14)
            fig.canvas.draw()
            current_idx[0] += 1
            
        except Exception as e:
            print(f"处理文件 {f} 出错: {e}")
            current_idx[0] += 1
            update()

    # 绑定交互事件
    def on_key(event):
        if event.key == 'escape':
            plt.close()
        elif event.key == ' ' or event.key == 'right':
            update()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update()
    plt.show()

if __name__ == '__main__':
    # 请确保路径指向你实际的 CSV
    display_bls_gsm_comparison('CBIS-DDSM/noise.csv')