import pydicom
import matplotlib.pyplot as plt
import random
from pathlib import Path
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import cv2  # 使用 OpenCV 处理图像非常快
import os
import pandas as pd
import concurrent.futures
from tqdm import tqdm  # 用于显示进度条

def process_dicom_pyramid(file_path, levels=3):
    # 1. 读取 DICOM 文件
    ds = pydicom.dcmread(file_path)
    img = ds.pixel_array.astype(np.float32)

    # 2. 预处理：归一化到 0-255 并转为 uint8 (OpenCV 常用格式)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. 构建高斯金字塔
    gaussian_pyramid = [img_norm]
    for i in range(levels):
        img_down = cv2.pyrDown(gaussian_pyramid[i])
        gaussian_pyramid.append(img_down)

    # 4. 构建拉普拉斯金字塔
    laplacian_pyramid = []
    for i in range(levels, 0, -1):
        # 将高层图像上采样
        size = (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0])
        img_up = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        
        # 计算差值：上一层高斯 - 当前层上采样
        laplacian = cv2.subtract(gaussian_pyramid[i-1], img_up)
        laplacian_pyramid.append(laplacian)

    return gaussian_pyramid, laplacian_pyramid

def display_pyramids(gp, lp):
    levels = len(gp)
    
    plt.figure(figsize=(12, 8))
    
    # 显示高斯金字塔 (Gaussian Pyramid)
    for i in range(levels):
        plt.subplot(2, levels, i + 1)
        plt.imshow(gp[i], cmap='gray')
        plt.title(f'Gaussian L{i}\n{gp[i].shape[:2]}')
        plt.axis('off')

    # 显示拉普拉斯金字塔 (Laplacian Pyramid)
    # 注意：lp[0] 对应的是最高分辨率的细节（由 GP[0] 和 GP[1] 产生）
    for i in range(len(lp)):
        plt.subplot(2, levels, i + 1 + levels)
        # 对拉普拉斯图像进行线性拉伸，方便观察细节
        lp_vis = cv2.normalize(lp[i], None, 0, 255, cv2.NORM_MINMAX)
        plt.imshow(lp_vis, cmap='gray')
        plt.title(f'Laplacian L{i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def display_top_noisy_dicoms(csv_path, num_files=127):
    # 1. 加载并筛选 CSV
    df = pd.read_csv(csv_path)
    
    # 确保按 NoiseScore 降序排列（噪声最大的在前）
    df_sorted = df.sort_values(by='NoiseScore', ascending=False)
    
    # 获取前 127 个文件路径
    top_noisy_files = df_sorted['FullPath'].head(num_files).tolist()
    scores = df_sorted['NoiseScore'].head(num_files).tolist()

    print(f"准备显示噪声最大的 {len(top_noisy_files)} 张图像。")
    print("操作说明：窗口激活状态下，按下键盘【任意键】切换下一张，关闭窗口退出。")

    # 2. 设置交互式显示函数
    current_idx = [0]  # 使用列表使内部函数可修改外部变量

    def show_next_image():
        if current_idx[0] >= len(top_noisy_files):
            print("已经看完所有选定文件。")
            plt.close()
            return

        file_path = top_noisy_files[current_idx[0]]
        score = scores[current_idx[0]]
        gp, lp = process_dicom_pyramid(file_path, levels=3)
        display_pyramids(gp, lp)
       
        
        try:
            # 读取并处理图像
            ds = pydicom.dcmread(file_path)
            # 应用窗技（VOI LUT）以获得正常对比度
            image = apply_voi_lut(ds.pixel_array, ds)
            
            plt.clf() # 清除当前画布
            plt.imshow(image, cmap='gray')
            plt.title(f"Rank: {current_idx[0] + 1} / {num_files}\nScore: {score:.4f}\nFile: {Path(file_path).name}")
            plt.axis('off')
            plt.draw() # 重新绘图
            
            current_idx[0] += 1
        except Exception as e:
            print(f"读取失败 {file_path}: {e}")
            current_idx[0] += 1
            show_next_image()

    def on_key(event):
        # 只要按下键盘，就切换下一张
        show_next_image()

    # 3. 初始化绘图窗口
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # 显示第一张
    show_next_image()
    plt.show()


# def calculate_noise_score(dcm_path):
#     """
#     快速估算单个 DICOM 的噪声强度
#     """
#     try:
#         # stop_before_pixels=False 必须为 False 才能读取图像数据
#         ds = pydicom.dcmread(dcm_path)
#         img = ds.pixel_array.astype(np.float32)

#         # 采样优化：只取中间区域，避免背景干扰，并提升速度
#         h, w = img.shape
#         roi = img[h//4:3*h//4, w//4:3*w//4]

#         # 拉普拉斯方差法
#         score = cv2.Laplacian(roi, cv2.CV_32F).var()
        
#         return str(dcm_path.name), score, str(dcm_path)
#     except Exception as e:
#         return str(dcm_path.name), None, str(dcm_path)


def calculate_noise_score(dcm_path):
    try:
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(np.float32)
        
        # 1. 归一化到 0-1
        img_min, img_max = np.min(img), np.max(img)
        if img_max - img_min == 0: return str(dcm_path.name), 0.0, str(dcm_path)
        img = (img - img_min) / (img_max - img_min)
        
        # 2. 采样中心区域，避免边缘伪影
        h, w = img.shape
        roi = img[h//4:3*h//4, w//4:3*w//4]
        
        # 3. 生成掩码：只关注有组织的部分（像素值大于 0.05 的区域）
        # 这样可以彻底排除纯黑背景导致的 0 分问题
        mask = roi > 0.05 
        if np.sum(mask) < 100: # 如果该区域全是背景
            return str(dcm_path.name), 0.0, str(dcm_path)

        # 4. 计算局部标准差图
        ksize = 7
        mean = cv2.blur(roi, (ksize, ksize))
        mean_sq = cv2.blur(roi**2, (ksize, ksize))
        var_map = np.maximum(0, mean_sq - mean**2)
        std_map = np.sqrt(var_map)
        
        # 5. 只提取掩码区域内的标准差值
        valid_std_values = std_map[mask]
        
        # 6. 在“组织内”寻找相对平滑的区域
        # 选组织内平滑度排在 10% 到 50% 之间的值（避开边缘和纯黑背景）
        low_val = np.percentile(valid_std_values, 10)
        high_val = np.percentile(valid_std_values, 50)
        
        # 取这个区间内的均值作为噪声分
        score = np.mean(valid_std_values[(valid_std_values >= low_val) & (valid_std_values <= high_val)])
        
        return str(dcm_path.name), float(score), str(dcm_path)
        
    except Exception as e:
        return str(dcm_path.name), None, str(dcm_path)

# def calculate_noise_score(dcm_path):
#     """
#     估算图像噪声强度。
#     得分越高，代表噪声可能越强。
#     """
#     ds = pydicom.dcmread(dcm_path)
#     img = ds.pixel_array.astype(np.float32)
    
#     # 归一化到 0-255 以便统一评估标准
#     img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    
#     # 方法：计算原图与中值滤波图的差值（即提取出的噪声分量）
#     # 中值滤波能很好地保留边缘并抑制孤立噪声点
#     denoised = cv2.medianBlur(img, 3)
#     noise_component = img - denoised
    
#     # 计算噪声分量的方差作为得分
#     score = np.var(noise_component)
#     return score

def sort_dicoms_by_noise(folder_path):
    folder = Path(folder_path)
    # 获取所有 DICOM 文件
    dcm_files = list(folder.rglob("*.dcm")) + list(folder.rglob("*.dicom"))
    print(f"开始处理 {len(dcm_files)} 个文件...")

    results = []
    
    # 使用进程池并行处理 (max_workers 默认使用 CPU 核心数)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 使用 tqdm 显示实时进度
        futures = [executor.submit(calculate_noise_score, f) for f in dcm_files]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            filename, score, full_path = future.result()
            if score is not None:
                results.append({
                    'FileName': filename,
                    'NoiseScore': score,
                    'FullPath': full_path
                })

    # 1. 转换为 DataFrame
    df = pd.DataFrame(results)

    # 2. 按噪声强度排序
    df = df.sort_values(by='NoiseScore', ascending=False).reset_index(drop=True)

    # 3. 保存为 CSV
    output_csv = os.path.join(folder_path, 'noise.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n处理完成！")
    print(f"结果已保存至: {output_csv}")
    print("\n前 127 个噪声最大的文件：")
    print(df.head(127))


# def estimate_noise_score(dcm_path):
#     ds = pydicom.dcmread(dcm_path)
#     img = ds.pixel_array
    
#     # 提速技巧 1：降采样（取 1/4 大小的图像）
#     # 使用切片操作比 cv2.resize 快得多
#     img_small = img[::2, ::2] 
    
#     # 提速技巧 2：只计算中心区域（避开四周大量黑色背景）
#     h, w = img_small.shape
#     center_roi = img_small[h//4:3*h//4, w//4:3*w//4]
    
#     # 计算拉普拉斯方差
#     return cv2.Laplacian(center_roi.astype(np.float32), cv2.CV_32F).var()

# def sort_dicoms_by_noise(folder_path):
#     folder = Path(folder_path)
#     extensions = ['*.dcm', '*.dicom', '*.DCM', '*.DICOM']
#     dcm_files = []

#     for ext in extensions:
#         dcm_files.extend(folder.rglob(ext))
    
#     results = []
#     print(f"正在分析 {len(dcm_files)} 个文件的噪声水平...")
    
#     for f in tqdm(dcm_files):
#         try:
#             score = estimate_noise_score(f)
#             # print('path', f)
#             # print('score', score)
#             results.append({'path': f, 'score': score})
#         except Exception as e:
#             print(f"处理失败 {f.name}: {e}")
            
#     # 按得分从低到高排序（噪声最小的排在前面）
#     sorted_results = sorted(results, key=lambda x: x['score'])
    
#     return sorted_results


def read_dicom(folder_path):
    folder = Path(folder_path)
    extensions = ['*.dcm', '*.dicom', '*.DCM', '*.DICOM']
    all_dicom_files = []

    for ext in extensions:
        all_dicom_files.extend(folder.rglob(ext))

    # 转换为字符串列表
    file_list = [str(f) for f in all_dicom_files]
    print(f"找到 {len(file_list)} 个 DICOM 文件")
    
    if not file_list:
        print("未找到任何 DICOM 文件，请检查路径。")
        return

    # --- 随机选择一个并显示 ---
    
    # 1. 随机选取一个路径
    random_file = random.choice(file_list)
    print(f"正在读取随机选取的图像: {random_file}")

    # 2. 使用 pydicom 读取
    ds = pydicom.dcmread(random_file)

    # 3. 提取像素数据
    # 对于 VinDr-Mammo 数据集，建议使用 apply_voi_lut 处理窗宽窗位
    # 否则图像可能看起来是一片漆黑或对比度极低
    image = ds.pixel_array
    if 'WindowCenter' in ds:
        image = apply_voi_lut(ds.pixel_array, ds)

    # 4. 绘图显示
    plt.figure(figsize=(8, 8))
    # 乳腺影像通常使用 gray 或 bone 颜色映射
    plt.imshow(image, cmap='gray') 
    plt.title(f"DICOM Image: {Path(random_file).name}\nSize: {image.shape}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # # 确保文件夹路径正确
    # folder = 'vindr-mammo'
    # read_dicom(folder)
    # --- 执行 ---
    # folder = 'CBIS-DDSM'
    # sorted_list = sort_dicoms_by_noise(folder)
    CSV_FILE = 'CBIS-DDSM/noise.csv'
    display_top_noisy_dicoms(CSV_FILE, num_files=127)

