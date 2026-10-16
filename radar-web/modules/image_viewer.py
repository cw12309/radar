"""
影像查看器模块
实现2D切片查看（三视图）和基础交互功能
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib


class ImageViewer:
    """医学影像2D切片查看器"""
    
    # CT窗宽窗位预设
    WINDOW_PRESETS = {
        '腹部': {'center': 40, 'width': 400},
        '肺窗': {'center': -600, 'width': 1500},
        '骨窗': {'center': 400, 'width': 1800},
        '脑窗': {'center': 40, 'width': 80},
        '纵隔': {'center': 50, 'width': 350},
    }
    
    def __init__(self, image_array, affine=None, spacing=None):
        """初始化查看器
        
        Args:
            image_array: 3D影像数组 (H, W, D)
            affine: 仿射矩阵
            spacing: 体素间距 (x, y, z)
        """
        self.image_array = image_array
        self.affine = affine
        self.spacing = spacing or (1.0, 1.0, 1.0)
        
        # 获取影像尺寸
        self.shape = image_array.shape
        
        # 默认切片位置（中心）
        self.axial_idx = self.shape[2] // 2      # Z轴
        self.sagittal_idx = self.shape[0] // 2   # X轴
        self.coronal_idx = self.shape[1] // 2    # Y轴
        
        # 默认窗宽窗位
        self.window_center = 40
        self.window_width = 400
        
    def apply_window(self, image_slice, center=None, width=None):
        """应用窗宽窗位
        
        Args:
            image_slice: 2D影像切片
            center: 窗位中心
            width: 窗宽
            
        Returns:
            np.ndarray: 窗口调整后的切片
        """
        center = center or self.window_center
        width = width or self.window_width
        
        lower = center - width / 2
        upper = center + width / 2
        
        windowed = np.clip(image_slice, lower, upper)
        windowed = (windowed - lower) / (upper - lower)
        
        return windowed
    
    def get_axial_slice(self, slice_idx=None, apply_window_level=True):
        """获取轴位切片（Axial - 横断面）
        
        Args:
            slice_idx: 切片索引，None则使用当前位置
            apply_window_level: 是否应用窗宽窗位
            
        Returns:
            np.ndarray: 2D切片
        """
        idx = slice_idx if slice_idx is not None else self.axial_idx
        idx = np.clip(idx, 0, self.shape[2] - 1)
        
        slice_data = self.image_array[:, :, idx]
        
        if apply_window_level:
            slice_data = self.apply_window(slice_data)
        
        return slice_data
    
    def get_sagittal_slice(self, slice_idx=None, apply_window_level=True):
        """获取矢状位切片（Sagittal - 纵断面）
        
        Args:
            slice_idx: 切片索引
            apply_window_level: 是否应用窗宽窗位
            
        Returns:
            np.ndarray: 2D切片
        """
        idx = slice_idx if slice_idx is not None else self.sagittal_idx
        idx = np.clip(idx, 0, self.shape[0] - 1)
        
        slice_data = self.image_array[idx, :, :]
        
        if apply_window_level:
            slice_data = self.apply_window(slice_data)
        
        return slice_data
    
    def get_coronal_slice(self, slice_idx=None, apply_window_level=True):
        """获取冠状位切片（Coronal - 冠状面）
        
        Args:
            slice_idx: 切片索引
            apply_window_level: 是否应用窗宽窗位
            
        Returns:
            np.ndarray: 2D切片
        """
        idx = slice_idx if slice_idx is not None else self.coronal_idx
        idx = np.clip(idx, 0, self.shape[1] - 1)
        
        slice_data = self.image_array[:, idx, :]
        
        if apply_window_level:
            slice_data = self.apply_window(slice_data)
        
        return slice_data
    
    def set_window_preset(self, preset_name):
        """设置窗宽窗位预设
        
        Args:
            preset_name: 预设名称
        """
        if preset_name in self.WINDOW_PRESETS:
            preset = self.WINDOW_PRESETS[preset_name]
            self.window_center = preset['center']
            self.window_width = preset['width']
    
    def overlay_mask(self, image_slice, mask_slice, alpha=0.3, colormap='rainbow', 
                    specific_labels=None, label_colors=None):
        """在影像切片上叠加分割mask
        
        Args:
            image_slice: 原始影像切片
            mask_slice: 分割mask切片（可以是多标签）
            alpha: 透明度
            colormap: 颜色映射
            specific_labels: 要显示的特定标签列表（None表示全部显示）
            label_colors: 标签颜色字典 {label: color}
            
        Returns:
            np.ndarray: RGB图像
        """
        # 归一化影像到[0, 1]
        if image_slice.max() > 1:
            image_norm = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
        else:
            image_norm = image_slice
        
        # 转换为RGB
        image_rgb = np.stack([image_norm] * 3, axis=-1)
        
        # 如果有mask，叠加显示
        if mask_slice is not None and mask_slice.max() > 0:
            # 如果指定了特定标签
            if specific_labels is not None:
                display_mask = np.zeros_like(mask_slice)
                for label in specific_labels:
                    display_mask[mask_slice == label] = label
                mask_slice = display_mask
            
            # 使用自定义颜色或colormap
            if label_colors is not None:
                # 为每个标签应用特定颜色
                for label, color in label_colors.items():
                    mask_binary = mask_slice == label
                    if mask_binary.any():
                        # 解析颜色（支持hex或RGB）
                        if isinstance(color, str) and color.startswith('#'):
                            # hex颜色转RGB
                            color_rgb = tuple(int(color[i:i+2], 16)/255.0 for i in (1, 3, 5))
                        else:
                            color_rgb = color
                        
                        # 叠加颜色
                        for c in range(3):
                            image_rgb[mask_binary, c] = (1 - alpha) * image_rgb[mask_binary, c] + alpha * color_rgb[c]
            else:
                # 使用colormap
                cmap = plt.get_cmap(colormap)
                if mask_slice.max() > 1:
                    mask_norm = mask_slice / mask_slice.max()
                else:
                    mask_norm = mask_slice
                mask_rgb = cmap(mask_norm)[..., :3]
                
                # 只在有mask的地方叠加
                mask_binary = mask_slice > 0
                image_rgb[mask_binary] = (1 - alpha) * image_rgb[mask_binary] + alpha * mask_rgb[mask_binary]
        
        return image_rgb
    
    def create_three_view_figure(self, mask_array=None, figsize=(15, 5)):
        """创建三视图显示
        
        Args:
            mask_array: 可选的分割mask数组
            figsize: 图像尺寸
            
        Returns:
            matplotlib.figure.Figure: 图像对象
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 轴位
        axial = self.get_axial_slice()
        if mask_array is not None:
            mask_axial = mask_array[:, :, self.axial_idx]
            axial = self.overlay_mask(axial, mask_axial)
        axes[0].imshow(axial, cmap='gray' if mask_array is None else None, origin='lower')
        axes[0].set_title(f'轴位 (Axial) - 切片 {self.axial_idx}/{self.shape[2]}')
        axes[0].axis('off')
        
        # 矢状位
        sagittal = self.get_sagittal_slice()
        if mask_array is not None:
            mask_sagittal = mask_array[self.sagittal_idx, :, :]
            sagittal = self.overlay_mask(sagittal, mask_sagittal)
        axes[1].imshow(sagittal, cmap='gray' if mask_array is None else None, origin='lower')
        axes[1].set_title(f'矢状位 (Sagittal) - 切片 {self.sagittal_idx}/{self.shape[0]}')
        axes[1].axis('off')
        
        # 冠状位
        coronal = self.get_coronal_slice()
        if mask_array is not None:
            mask_coronal = mask_array[:, self.coronal_idx, :]
            coronal = self.overlay_mask(coronal, mask_coronal)
        axes[2].imshow(coronal, cmap='gray' if mask_array is None else None, origin='lower')
        axes[2].set_title(f'冠状位 (Coronal) - 切片 {self.coronal_idx}/{self.shape[1]}')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def get_slice_range(self, view='axial'):
        """获取指定视图的切片范围
        
        Args:
            view: 视图类型 ('axial', 'sagittal', 'coronal')
            
        Returns:
            tuple: (min_idx, max_idx)
        """
        if view == 'axial':
            return (0, self.shape[2] - 1)
        elif view == 'sagittal':
            return (0, self.shape[0] - 1)
        elif view == 'coronal':
            return (0, self.shape[1] - 1)
        else:
            raise ValueError(f"未知的视图类型: {view}")


def load_image_for_viewer(nifti_path):
    """从NIfTI文件加载影像用于查看器
    
    Args:
        nifti_path: NIfTI文件路径
        
    Returns:
        ImageViewer: 查看器对象
    """
    nifti_img = nib.load(nifti_path)
    image_array = nifti_img.get_fdata()
    affine = nifti_img.affine
    
    # 计算体素间距
    spacing = (
        abs(affine[0, 0]),
        abs(affine[1, 1]),
        abs(affine[2, 2])
    )
    
    return ImageViewer(image_array, affine, spacing)


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    test_image = np.random.randn(256, 256, 100) * 100 + 40
    viewer = ImageViewer(test_image)
    
    print(f"影像尺寸: {viewer.shape}")
    print(f"轴位切片范围: {viewer.get_slice_range('axial')}")
    
    # 测试获取切片
    axial = viewer.get_axial_slice()
    print(f"轴位切片形状: {axial.shape}")
