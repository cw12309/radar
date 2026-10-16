"""
病灶定位工具
自动定位病灶区域并提供可视化支持
"""
import numpy as np
from typing import List, Dict, Tuple, Optional


class LesionLocalizer:
    """病灶定位器"""
    
    def __init__(self):
        pass
    
    def find_lesion_center(self, mask_array: np.ndarray, label: int = 1) -> Tuple[int, int, int]:
        """查找病灶中心点
        
        Args:
            mask_array: 3D mask数组
            label: 病灶标签值
            
        Returns:
            tuple: (z, y, x) 中心坐标
        """
        coords = np.where(mask_array == label)
        
        if len(coords[0]) == 0:
            return None
        
        center_z = int(np.mean(coords[0]))
        center_y = int(np.mean(coords[1]))
        center_x = int(np.mean(coords[2]))
        
        return (center_z, center_y, center_x)
    
    def get_bounding_box(self, mask_array: np.ndarray, label: int = 1) -> Dict:
        """获取病灶边界框
        
        Args:
            mask_array: 3D mask数组
            label: 病灶标签值
            
        Returns:
            dict: 边界框信息
        """
        coords = np.where(mask_array == label)
        
        if len(coords[0]) == 0:
            return None
        
        bbox = {
            'z_min': int(np.min(coords[0])),
            'z_max': int(np.max(coords[0])),
            'y_min': int(np.min(coords[1])),
            'y_max': int(np.max(coords[1])),
            'x_min': int(np.min(coords[2])),
            'x_max': int(np.max(coords[2])),
        }
        
        # 计算尺寸
        bbox['size_z'] = bbox['z_max'] - bbox['z_min'] + 1
        bbox['size_y'] = bbox['y_max'] - bbox['y_min'] + 1
        bbox['size_x'] = bbox['x_max'] - bbox['x_min'] + 1
        bbox['volume'] = bbox['size_z'] * bbox['size_y'] * bbox['size_x']
        
        # 计算中心
        bbox['center_z'] = (bbox['z_min'] + bbox['z_max']) // 2
        bbox['center_y'] = (bbox['y_min'] + bbox['y_max']) // 2
        bbox['center_x'] = (bbox['x_min'] + bbox['x_max']) // 2
        
        return bbox
    
    def find_organ_lesions(self, 
                          diagnosis_result: Dict,
                          organ_name: str) -> List[Dict]:
        """查找特定器官的所有病灶
        
        Args:
            diagnosis_result: 诊断结果字典
            organ_name: 器官名称
            
        Returns:
            list: 病灶信息列表
        """
        lesions = []
        
        diagnoses = diagnosis_result.get('diagnoses', [])
        for diag in diagnoses:
            if diag.get('organ') == organ_name:
                lesion_info = {
                    'finding': diag.get('finding'),
                    'confidence': diag.get('confidence'),
                    'severity': diag.get('severity'),
                    'organ': organ_name
                }
                lesions.append(lesion_info)
        
        return lesions
    
    def get_optimal_view_slice(self, 
                               mask_array: np.ndarray,
                               label: int = 1,
                               axis: str = 'axial') -> int:
        """获取病灶最佳查看切片
        
        Args:
            mask_array: 3D mask数组
            label: 病灶标签值
            axis: 视图轴 ('axial', 'sagittal', 'coronal')
            
        Returns:
            int: 最佳切片索引
        """
        # 找到中心点
        center = self.find_lesion_center(mask_array, label)
        
        if center is None:
            # 如果没找到，返回中间切片
            if axis == 'axial':
                return mask_array.shape[2] // 2
            elif axis == 'sagittal':
                return mask_array.shape[0] // 2
            else:  # coronal
                return mask_array.shape[1] // 2
        
        # 根据轴返回对应索引
        if axis == 'axial':
            return center[0]  # z轴
        elif axis == 'sagittal':
            return center[2]  # x轴
        else:  # coronal
            return center[1]  # y轴
    
    def calculate_lesion_statistics(self, mask_array: np.ndarray, label: int = 1) -> Dict:
        """计算病灶统计信息
        
        Args:
            mask_array: 3D mask数组
            label: 病灶标签值
            
        Returns:
            dict: 统计信息
        """
        voxel_count = np.sum(mask_array == label)
        
        if voxel_count == 0:
            return None
        
        # 获取边界框
        bbox = self.get_bounding_box(mask_array, label)
        
        stats = {
            'voxel_count': int(voxel_count),
            'bounding_box': bbox,
            'extent_ratio': voxel_count / bbox['volume'] if bbox['volume'] > 0 else 0,
        }
        
        return stats
    
    def create_highlighted_view(self,
                               image_slice: np.ndarray,
                               mask_slice: np.ndarray,
                               lesion_label: int = 1,
                               highlight_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
        """创建病灶高亮视图
        
        Args:
            image_slice: 原始影像切片
            mask_slice: mask切片
            lesion_label: 病灶标签
            highlight_color: 高亮颜色 (R, G, B)
            
        Returns:
            np.ndarray: RGB高亮图像
        """
        # 归一化影像
        if image_slice.max() > 1:
            image_norm = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
        else:
            image_norm = image_slice
        
        # 转换为RGB
        image_rgb = np.stack([image_norm] * 3, axis=-1)
        
        # 高亮病灶区域
        lesion_mask = mask_slice == lesion_label
        if lesion_mask.any():
            alpha = 0.5
            for c in range(3):
                image_rgb[lesion_mask, c] = (1 - alpha) * image_rgb[lesion_mask, c] + alpha * highlight_color[c]
        
        return image_rgb
    
    def generate_lesion_report(self, 
                              lesions: List[Dict],
                              organ_name: str) -> str:
        """生成病灶报告
        
        Args:
            lesions: 病灶列表
            organ_name: 器官名称
            
        Returns:
            str: 病灶报告文本
        """
        if not lesions:
            return f"{organ_name}: 未发现异常"
        
        report = f"## {organ_name} 病灶报告\n\n"
        report += f"发现 {len(lesions)} 处异常:\n\n"
        
        for i, lesion in enumerate(lesions, 1):
            report += f"### {i}. {lesion['finding']}\n"
            report += f"- 置信度: {lesion['confidence']:.1%}\n"
            report += f"- 严重程度: {lesion['severity']}\n\n"
        
        return report


# 测试代码
if __name__ == "__main__":
    localizer = LesionLocalizer()
    
    # 创建测试mask
    test_mask = np.zeros((100, 100, 100), dtype=np.uint8)
    test_mask[40:50, 40:50, 40:50] = 1  # 模拟病灶
    
    center = localizer.find_lesion_center(test_mask, 1)
    print(f"病灶中心: {center}")
    
    bbox = localizer.get_bounding_box(test_mask, 1)
    print(f"边界框: {bbox}")
    
    stats = localizer.calculate_lesion_statistics(test_mask, 1)
    print(f"统计信息: {stats}")
