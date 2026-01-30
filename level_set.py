import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd
import pydicom
import random
from skimage.filters import gabor
from matplotlib.widgets import RectangleSelector



class SBGFRLS_Segmentor:
    def __init__(self, image, alpha=10, sigma=1.5):
        self.I = image.astype(float)
        self.alpha = alpha  # 演化步长
        self.sigma = sigma  # 高斯平滑的标准差
        self.phi = None
        
    def initialize_phi(self, roi_rect):
        """初始化水平集函数: 矩形内为-1, 矩形外为1"""
        self.phi = np.ones(self.I.shape)
        x1, y1, x2, y2 = roi_rect
        self.phi[y1:y2, x1:x2] = -1
        
    def get_spf(self):
        """计算论文核心：符号压力函数 (SPF)"""
        # 计算内外平均值
        c1 = np.sum(self.I * (self.phi < 0)) / (np.sum(self.phi < 0) + 1e-10)
        c2 = np.sum(self.I * (self.phi >= 0)) / (np.sum(self.phi >= 0) + 1e-10)
        
        # SPF 函数公式
        spf = (self.I - (c1 + c2)/2) / (np.max(np.abs(self.I)) + 1e-10)
        return spf

    def update(self, iterations=50):
        """算法迭代演化"""
        for _ in range(iterations):
            spf = self.get_spf()
            
            # 1. 根据演化方程更新 phi
            # 论文简化版：phi = phi + alpha * spf
            self.phi = self.phi + self.alpha * spf
            
            # 2. 核心步骤：选择性二值化 (Selective Binary)
            self.phi[self.phi > 0] = 1
            self.phi[self.phi < 0] = -1
            
            # 3. 核心步骤：高斯滤波正则化 (Gaussian Filtering)
            self.phi = gaussian_filter(self.phi, sigma=self.sigma)

# --- 交互界面部分 ---
def interactive_segmentation(dicom_path):
    # 读取 DICOM
    ds = sitk.ReadImage(dicom_path)
    img = sitk.GetArrayFromImage(ds)[0] # 取第一层
    
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title("Click and drag to select initial region")
    
    # 简单的矩形框选交互
    from matplotlib.widgets import RectangleSelector
    
    def onselect(eclick, erelease):
        roi = [int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)]
        
        # 运行算法
        seg = SBGFRLS_Segmentor(img)
        seg.initialize_phi(roi)
        
        print("Segmenting... please wait.")
        seg.update(iterations=200) # 迭代次数根据图像调整
        
        # 显示结果
        ax.contour(seg.phi, [0], colors='r', linewidths=2)
        plt.draw()
        print("Done!")

    rs = RectangleSelector(ax, onselect, drawtype='box', useblit=True,
                           button=[1], minspanx=5, minspany=5, spandata=True)
    plt.show()

# ... [SBGFRLS_Segmentor 类保持不变] ...

class RandomDicomViewer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.rs = None
        self.current_img = None
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        print(">>> 操作指南: \n1. 鼠标拖动框选 ROI 开始分割\n2. 按 'Any Key' (除ESC) 随机换一张图\n3. 按 'ESC' 退出程序")
        
        self.show_random_image()
        plt.show()

    def show_random_image(self):
            """随机抽取并显示一张图像"""
            self.ax.clear()
            
            random_row = self.df.sample(n=1).iloc[0]
            f_path = random_row['FullPath']
            
            try:
                print(f"Loading: {f_path}")
                ds = sitk.ReadImage(f_path)
                img_array = sitk.GetArrayFromImage(ds)
                # 处理医学图像常见的维度问题
                self.current_img = img_array[0] if img_array.ndim == 3 else img_array
                
                # 自动调整对比度（解决医学图像发黑问题）
                vmin, vmax = np.percentile(self.current_img, [2, 98])
                self.ax.imshow(self.current_img, cmap='gray', vmin=vmin, vmax=vmax)
                
                self.ax.set_title(f"Random DICOM: {random_row.get('NoiseScore', '')}\nDrag a box to start SBGFRLS")
                
                # 最新版 Matplotlib 兼容写法
                from matplotlib.widgets import RectangleSelector
                self.rs = RectangleSelector(
                    self.ax, 
                    self.on_select, 
                    useblit=True,
                    props=dict(edgecolor='red', linestyle='--', fill=False, linewidth=2)
                )
                self.fig.canvas.draw()
                
            except Exception as e:
                print(f"Error loading {f_path}: {e}")
                # 如果加载失败，继续尝试下一张
                self.show_random_image()

    def on_select(self, eclick, erelease):
        """当完成框选时的动作"""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        roi = [x1, y1, x2, y2]
        
        # 实例化算法
        seg = SBGFRLS_Segmentor(self.current_img)
        seg.initialize_phi(roi)
        
        print("Segmenting...")
        seg.update(iterations=200) # 也可以根据需要改为 300
        
        # 绘制红色等值线 (Level Set = 0)
        self.ax.contour(seg.phi, [0], colors='r', linewidths=2)
        self.ax.set_title("Segmentation Done! Press any key for next.")
        self.fig.canvas.draw()

    def on_key(self, event):
        """按键处理"""
        if event.key == 'escape':
            plt.close(self.fig)
        else:
            self.show_random_image()



class ChanVese_Segmentor:
    def __init__(self, image, mu=0.1, lambda1=1, lambda2=1, dt=0.5):
        self.I = image.astype(float)
        self.mu = mu          # 长度项系数（控制轮廓平滑度）
        self.l1 = lambda1    # 内部拟合项系数
        self.l2 = lambda2    # 外部拟合项系数
        self.dt = dt          # 时间步长
        self.phi = None

    def initialize_phi(self, roi_rect):
        """初始化水平集：内部为-1，外部为1"""
        self.phi = np.ones(self.I.shape)
        x1, y1, x2, y2 = roi_rect
        self.phi[y1:y2, x1:x2] = -1

    def update(self, iterations=100):
        """C-V 模型的偏微分方程演化"""
        for _ in range(iterations):
            # 1. 计算内外区域的平均值 c1, c2
            inside = self.phi < 0
            outside = self.phi >= 0
            
            c1 = np.sum(self.I * inside) / (np.sum(inside) + 1e-10)
            c2 = np.sum(self.I * outside) / (np.sum(outside) + 1e-10)

            # 2. 计算正则化项（曲率）- 这里简化处理，保证平滑
            # 在原始 C-V 中这步较复杂，这里采用高斯平滑代替以提高稳定性
            self.phi = gaussian_filter(self.phi, sigma=0.5)

            # 3. 演化方程核心：(I-c1)^2 - (I-c2)^2
            # 这里的逻辑是：如果像素更接近 c1，该项为负，phi 减小（保持内部状态）
            force = self.l1 * (self.I - c1)**2 - self.l2 * (self.I - c2)**2
            
            self.phi = self.phi - self.dt * force
            
            # 保持 phi 在合理范围内（防止数值爆炸）
            self.phi = np.clip(self.phi, -10, 10)

class CV_DicomViewer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.show_random_image()
        plt.show()

    def show_random_image(self):
        self.ax.clear()
        row = self.df.sample(n=1).iloc[0]
        f_path = row['FullPath']
        try:
            ds = sitk.ReadImage(f_path)
            img = sitk.GetArrayFromImage(ds)
            self.current_img = img[0] if img.ndim == 3 else img
            
            # 归一化显示对比度
            vmin, vmax = np.percentile(self.current_img, [1, 99])
            self.ax.imshow(self.current_img, cmap='gray', vmin=vmin, vmax=vmax)
            self.ax.set_title(f"Chan-Vese Model\nDrag box to start | ESC to exit")
            
            from matplotlib.widgets import RectangleSelector
            self.rs = RectangleSelector(self.ax, self.on_select, useblit=True,
                                       props=dict(edgecolor='cyan', fill=False))
            self.fig.canvas.draw()
        except:
            self.show_random_image()

    def on_select(self, eclick, erelease):
        roi = [int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)]
        seg = ChanVese_Segmentor(self.current_img)
        seg.initialize_phi(roi)
        
        print("C-V Segmenting...")
        seg.update(iterations=150)
        
        # 绘制 0 水平集（分割线）
        self.ax.contour(seg.phi, [0], colors='yellow', linewidths=2)
        self.ax.set_title("Done! Any key to next.")
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'escape': plt.close()
        else: self.show_random_image()



class GAC_Segmentor:
    def __init__(self, image, alpha=1.0, beta=0.1, dt=0.2):
        self.I = image.astype(float)
        self.alpha = alpha  
        self.beta = beta    
        self.dt = dt        
        self.phi = None
        
        # 1. 预计算边缘停止函数 g(I)
        # 先平滑去噪
        I_smooth = gaussian_filter(self.I, sigma=1.0)
        
        # 使用 numpy.gradient 代替 scipy
        iy, ix = np.gradient(I_smooth) 
        grad_mag = np.sqrt(ix**2 + iy**2)
        
        # 边缘停止函数: g = 1 / (1 + |grad|^2)
        # 这里对梯度做归一化处理，防止数值过大
        self.g = 1.0 / (1.0 + (grad_mag / (grad_mag.mean() + 1e-10))**2)
        
    def initialize_phi(self, roi_rect):
        self.phi = np.ones(self.I.shape)
        x1, y1, x2, y2 = roi_rect
        self.phi[y1:y2, x1:x2] = -1

    def update(self, iterations=300):
        for _ in range(iterations):
            # 使用 np.gradient 计算水平集函数 phi 的梯度
            dy, dx = np.gradient(self.phi)
            mag = np.sqrt(dx**2 + dy**2) + 1e-10
            
            # 计算单位法向量
            nx = dx / mag
            ny = dy / mag
            
            # 计算散度即曲率 (Curvature)
            _, nxx = np.gradient(nx) # 对 nx 求 x 方向导数
            nyy, _ = np.gradient(ny) # 对 ny 求 y 方向导数
            curvature = nxx + nyy
            
            # GAC 演化方程: d_phi/dt = g * (alpha + beta * curvature) * |grad_phi|
            force = self.g * (self.alpha + self.beta * curvature) * mag
            
            self.phi = self.phi + self.dt * force
            
            # 数值稳定性限制
            self.phi = np.clip(self.phi, -2, 2)

class GAC_DicomViewer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.show_random_image()
        plt.show()

    def show_random_image(self):
        self.ax.clear()
        row = self.df.sample(n=1).iloc[0]
        f_path = row['FullPath']
        try:
            ds = sitk.ReadImage(f_path)
            img = sitk.GetArrayFromImage(ds)
            self.current_img = img[0] if img.ndim == 3 else img
            
            vmin, vmax = np.percentile(self.current_img, [1, 99])
            self.ax.imshow(self.current_img, cmap='gray', vmin=vmin, vmax=vmax)
            self.ax.set_title(f"GAC Model (Edge-based)\nDrag box to Start | Any Key for Next")
            
            from matplotlib.widgets import RectangleSelector
            self.rs = RectangleSelector(self.ax, self.on_select, useblit=True,
                                       props=dict(edgecolor='green', fill=False))
            self.fig.canvas.draw()
        except:
            self.show_random_image()

    def on_select(self, eclick, erelease):
        roi = [int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)]
        seg = GAC_Segmentor(self.current_img, alpha=-1.0) # alpha < 0 表示向内收缩
        seg.initialize_phi(roi)
        
        print("GAC Segmenting...")
        seg.update(iterations=400)
        
        self.ax.contour(seg.phi, [0], colors='lime', linewidths=2)
        self.ax.set_title("GAC Done! (Green line)")
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'escape': plt.close()
        else: self.show_random_image()


class DRLSE_Segmentor:
    def __init__(self, image, lambd=5.0, mu=0.2, alpha=-3.0, epsilon=1.5, dt=1.0):
        """
        lambd: 长度项系数 (Length term)
        mu: 距离正则化项系数 (Distance regularization term)
        alpha: 面积项/气球力系数 (Area term)
        epsilon: Dirac 函数的宽度参数
        """
        self.I = image.astype(float)
        self.lambd = lambd
        self.mu = mu
        self.alpha = alpha
        self.epsilon = epsilon
        self.dt = dt
        self.phi = None
        
        # 预计算边缘停止函数 g
        I_smooth = gaussian_filter(self.I, sigma=1.5)
        iy, ix = np.gradient(I_smooth)
        grad_mag_sq = ix**2 + iy**2
        self.g = 1.0 / (1.0 + grad_mag_sq / (grad_mag_sq.mean() + 1e-10))
        
        # 预计算 g 的梯度，用于演化方程
        self.gy, self.gx = np.gradient(self.g)

    def initialize_phi(self, roi_rect, c0=2):
        """初始化为常数二值函数，Li 论文的一大优势是不需要 SDF 初始化"""
        self.phi = np.ones(self.I.shape) * c0
        x1, y1, x2, y2 = roi_rect
        self.phi[y1:y2, x1:x2] = -c0

    def dirac(self, z):
        """正则化的 Dirac 函数"""
        return (1/2/self.epsilon) * (1 + np.cos(np.pi * z / self.epsilon)) * (np.abs(z) <= self.epsilon)

    def update(self, iterations=100):
        for _ in range(iterations):
            # 1. 计算 phi 的梯度和模长
            phi_y, phi_x = np.gradient(self.phi)
            mag = np.sqrt(phi_x**2 + phi_y**2) + 1e-10
            
            # 2. 距离正则化项 (Distance Regularization)
            # 简化版：使用拉普拉斯算子近似
            from scipy.ndimage import laplace
            dist_reg = laplace(self.phi) - (phi_x/mag + phi_y/mag) # 简化的正则化逻辑
            
            # 3. 外部能量项 (Edge term)
            nx = phi_x / mag
            ny = phi_y / mag
            _, nxx = np.gradient(nx)
            nyy, _ = np.gradient(ny)
            curvature = nxx + nyy
            
            edge_term = self.gx * nx + self.gy * ny + self.g * curvature
            
            # 4. 面积项 (Area term / Balloon force)
            area_term = self.g
            
            # 5. 总演化方程
            d_phi = self.mu * dist_reg + \
                    self.lambd * self.dirac(self.phi) * edge_term + \
                    self.alpha * self.dirac(self.phi) * area_term
            
            self.phi = self.phi + self.dt * d_phi

class DRLSE_Viewer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.show_random_image()
        plt.show()

    def show_random_image(self):
        self.ax.clear()
        row = self.df.sample(n=1).iloc[0]
        f_path = row['FullPath']
        try:
            ds = sitk.ReadImage(f_path)
            img = sitk.GetArrayFromImage(ds)
            self.current_img = img[0] if img.ndim == 3 else img
            
            vmin, vmax = np.percentile(self.current_img, [2, 98])
            self.ax.imshow(self.current_img, cmap='gray', vmin=vmin, vmax=vmax)
            self.ax.set_title("DRLSE Model (Li 2005)\nNo Re-initialization | Drag to Segment")
            
            from matplotlib.widgets import RectangleSelector
            self.rs = RectangleSelector(self.ax, self.on_select, useblit=True,
                                       props=dict(edgecolor='magenta', fill=False, linewidth=2))
            self.fig.canvas.draw()
        except:
            self.show_random_image()

    def on_select(self, eclick, erelease):
        roi = [int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)]
        # alpha 为负代表向内收缩
        seg = DRLSE_Segmentor(self.current_img, alpha=-3.0, mu=0.2) 
        seg.initialize_phi(roi)
        
        print("DRLSE Segmenting (Large time step allowed)...")
        seg.update(iterations=300)
        
        self.ax.contour(seg.phi, [0], colors='magenta', linewidths=2)
        self.ax.set_title("DRLSE Done! (Magenta line)")
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'escape': plt.close()
        else: self.show_random_image()


class Malladi_1995_Segmentor:
    def __init__(self, image, nu=1.0, theta=1.5, dt=0.1):
        """
        nu: 传播项系数 (气球力)，正值扩张，负值收缩
        theta: 停止函数的敏感度
        dt: 时间步长 (Malladi 模型对步长较敏感，建议设小一点)
        """
        self.I = image.astype(float)
        self.nu = nu
        self.dt = dt
        self.phi = None
        
        # 1. 预计算 Malladi 论文中的核心：停止函数 g(I)
        # 先进行高斯平滑以减少噪声干扰
        I_smooth = gaussian_filter(self.I, sigma=2.0)
        iy, ix = np.gradient(I_smooth)
        grad_mag = np.sqrt(ix**2 + iy**2)
        
        # 论文中的 g(I) 形式：使用指数衰减或反比例
        # 这里使用经典形式：g = 1 / (1 + |grad/theta|^2)
        self.g = 1.0 / (1.0 + (grad_mag / theta)**2)
        
    def initialize_phi(self, roi_rect):
        """初始化水平集：内部为-1，外部为1"""
        self.phi = np.ones(self.I.shape)
        x1, y1, x2, y2 = roi_rect
        self.phi[y1:y2, x1:x2] = -1

    def update(self, iterations=200):
            for i in range(iterations): # 注意这里把 _ 改成 i 避免歧义
                # 1. 计算梯度
                phi_y, phi_x = np.gradient(self.phi)
                mag = np.sqrt(phi_x**2 + phi_y**2) + 1e-10
                
                # 2. 计算单位法向量
                nx = phi_x / mag
                ny = phi_y / mag
                
                # 3. 计算曲率 (Divergence of normalized gradient)
                _, nxx = np.gradient(nx)
                nyy, _ = np.gradient(ny)
                curvature = nxx + nyy
                
                # 4. Malladi 演化方程: d_phi = g * (K + nu) * |grad_phi|
                # 修正：直接相乘，不要在 if 中判断数组
                d_phi = self.g * (curvature + self.nu) * mag
                self.phi = self.phi + self.dt * d_phi
                
                # 5. 数值稳定性：定期限制范围，而不是用 np.sign
                # np.sign 会把 phi 变成 -1, 0, 1，这会导致梯度消失
                # 建议使用 clip 保持数值连续性
                if i % 20 == 0:
                    self.phi = np.clip(self.phi, -2, 2)

class Malladi_Viewer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.show_random_image()
        plt.show()

    def show_random_image(self):
        self.ax.clear()
        row = self.df.sample(n=1).iloc[0]
        f_path = row['FullPath']
        try:
            ds = sitk.ReadImage(f_path)
            img = sitk.GetArrayFromImage(ds)
            self.current_img = img[0] if img.ndim == 3 else img
            
            vmin, vmax = np.percentile(self.current_img, [1, 99])
            self.ax.imshow(self.current_img, cmap='gray', vmin=vmin, vmax=vmax)
            self.ax.set_title("Malladi et al. 1995 (Front Propagation)\nRed line = segmented boundary")
            
            from matplotlib.widgets import RectangleSelector
            self.rs = RectangleSelector(self.ax, self.on_select, useblit=True,
                                       props=dict(edgecolor='red', fill=False, linewidth=2))
            self.fig.canvas.draw()
        except:
            self.show_random_image()

    def on_select(self, eclick, erelease):
            roi = [int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)]
            
            # 建议尝试：增大 theta 让它不容易被噪声卡住，增大 iterations 让它跑得更远
            # 如果你画的是小框，nu 用 2.0 跑快一点
            seg = Malladi_1995_Segmentor(self.current_img, nu=2.0, theta=10.0, dt=0.2)
            seg.initialize_phi(roi)
            
            print("Malladi Front Propagating... (More iterations for visible movement)")
            seg.update(iterations=800) # 增加迭代次数
            
            self.ax.contour(seg.phi, [0], colors='red', linewidths=2)
            self.ax.set_title("Done! Red=Result, Blue=Initial")
            self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'escape': plt.close()
        else: self.show_random_image()


class Paragios_Deriche_Segmentor:
    def __init__(self, image, alpha=2.0, beta=0.5, dt=0.2):
        """
        alpha: 区域概率力的权重 (论文核心)
        beta: 曲率平滑项的权重
        dt: 时间步长
        """
        self.I = image.astype(float)
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.phi = None
        
        # 1. 纹理特征提取：模拟论文中的多通道滤波
        # 我们提取一个水平方向和一个垂直方向的纹理特征
        filt_real1, _ = gabor(self.I, frequency=0.1, theta=0)
        filt_real2, _ = gabor(self.I, frequency=0.1, theta=np.pi/2)
        # 特征融合：这里使用模长作为简化的纹理特征
        self.feature_map = np.sqrt(filt_real1**2 + filt_real2**2)
        
    def estimate_statistics(self, roi_rect):
        """
        监督学习：计算用户选定区域的概率分布参数 (PDF)
        """
        x1, y1, x2, y2 = roi_rect
        # 提取目标区域特征
        roi_features = self.feature_map[y1:y2, x1:x2]
        self.mu_obj = np.mean(roi_features)
        self.std_obj = np.std(roi_features) + 1e-5
        
        # 提取背景区域特征（简单处理：全图作为背景参考）
        self.mu_bg = np.mean(self.feature_map)
        self.std_bg = np.std(self.feature_map) + 1e-5

    def initialize_phi(self, roi_rect):
        # 初始化 Level Set 函数 (内负外正)
        self.phi = np.ones(self.I.shape)
        x1, y1, x2, y2 = roi_rect
        self.phi[y1:y2, x1:x2] = -1

    def update(self, iterations=200, ax_render=None, fig_render=None):
            # 计算区域概率力
            term_obj = -0.5 * ((self.feature_map - self.mu_obj) / self.std_obj)**2 - np.log(self.std_obj)
            term_bg = -0.5 * ((self.feature_map - self.mu_bg) / self.std_bg)**2 - np.log(self.std_bg)
            probability_force = term_obj - term_bg

            for i in range(iterations):
                phi_y, phi_x = np.gradient(self.phi)
                mag = np.sqrt(phi_x**2 + phi_y**2) + 1e-10
                
                # 曲率 K
                nx, ny = phi_x/mag, phi_y/mag
                _, nxx = np.gradient(nx)
                nyy, _ = np.gradient(ny)
                curvature = nxx + nyy
                
                # 演化方程：增加 alpha 权重，减小 beta
                d_phi = (self.alpha * probability_force + self.beta * curvature) * mag
                self.phi += self.dt * d_phi
                
                # 每隔 40 次迭代刷新一次界面，让你看到轮廓在动
                if i % 40 == 0:
                    self.phi = np.clip(self.phi, -2, 2)
                    if ax_render:
                        # 清除之前的轮廓，画新的
                        if hasattr(self, 'temp_contour'):
                            for coll in self.temp_contour.collections:
                                coll.remove()
                        self.temp_contour = ax_render.contour(self.phi, [0], colors='yellow', linewidths=1)
                        plt.pause(0.001) # 强制刷新

class TextureViewer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.show_random_image()
        plt.show()

    def show_random_image(self):
        self.ax1.clear()
        self.ax2.clear()
        row = self.df.sample(n=1).iloc[0]
        try:
            ds = sitk.ReadImage(row['FullPath'])
            img = sitk.GetArrayFromImage(ds)
            self.current_img = img[0] if img.ndim == 3 else img
            
            # 显示原图
            self.ax1.imshow(self.current_img, cmap='gray')
            self.ax1.set_title("Original DICOM - Drag a Box")
            
            # 预处理并展示纹理特征图
            self.seg_engine = Paragios_Deriche_Segmentor(self.current_img)
            self.ax2.imshow(self.seg_engine.feature_map, cmap='jet')
            self.ax2.set_title("Texture Feature Map (Gabor)")
            
            self.rs = RectangleSelector(self.ax1, self.on_select, useblit=True,
                                       props=dict(edgecolor='white', fill=False, linewidth=2))
            self.fig.canvas.draw()
        except Exception as e:
            print(f"Error: {e}")
            self.show_random_image()

    def on_select(self, eclick, erelease):
            roi = [int(min(eclick.xdata, erelease.xdata)), int(min(eclick.ydata, erelease.ydata)),
                int(max(eclick.xdata, erelease.xdata)), int(max(eclick.ydata, erelease.ydata))]
            
            # 调大 alpha (5.0)，调小 beta (0.1)
            self.seg_engine.alpha = 5.0 
            self.seg_engine.beta = 0.1
            
            self.seg_engine.estimate_statistics(roi)
            self.seg_engine.initialize_phi(roi)
            
            # 传入坐标轴对象以实现动画
            self.seg_engine.update(iterations=600, ax_render=self.ax1)
            self.fig.canvas.draw()
            
    def on_key(self, event):
        self.show_random_image()

if __name__ == '__main__':
    # TextureViewer('CBIS-DDSM/noise.csv')
    # Malladi_Viewer('CBIS-DDSM/noise.csv')
    # DRLSE_Viewer('CBIS-DDSM/noise.csv')
    # GAC_DicomViewer('CBIS-DDSM/noise.csv')
    # CV_DicomViewer('CBIS-DDSM/noise.csv')
    RandomDicomViewer('CBIS-DDSM/noise.csv')

