"""
影像对比工具
支持多时间点影像的并排对比和差异分析
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime


class ImageComparator:
    """影像对比器"""
    
    def __init__(self):
        pass
    
    def compare_two_studies(self,
                           study1: Dict,
                           study2: Dict,
                           slice_indices: Optional[Dict] = None) -> Dict:
        """对比两次检查
        
        Args:
            study1: 第一次检查 {'image': array, 'date': str, 'diagnoses': list}
            study2: 第二次检查
            slice_indices: 切片索引 {'axial': int, 'sagittal': int, 'coronal': int}
            
        Returns:
            dict: 对比结果
        """
        comparison = {
            'study1_date': study1.get('date'),
            'study2_date': study2.get('date'),
            'time_interval_days': self._calculate_interval(
                study1.get('date'), 
                study2.get('date')
            ),
            'diagnosis_changes': self._compare_diagnoses(
                study1.get('diagnoses', []),
                study2.get('diagnoses', [])
            )
        }
        
        return comparison
    
    def _calculate_interval(self, date1: str, date2: str) -> Optional[int]:
        """计算时间间隔（天数）"""
        try:
            d1 = datetime.fromisoformat(date1)
            d2 = datetime.fromisoformat(date2)
            return abs((d2 - d1).days)
        except:
            return None
    
    def _compare_diagnoses(self, diag1: List[Dict], diag2: List[Dict]) -> Dict:
        """对比诊断结果
        
        Returns:
            dict: {'new': [...], 'resolved': [...], 'persisting': [...], 'changed': [...]}
        """
        # 构建诊断标识 (organ + finding)
        def make_key(d):
            return f"{d.get('organ', '')}_{d.get('finding', '')}"
        
        set1 = {make_key(d): d for d in diag1}
        set2 = {make_key(d): d for d in diag2}
        
        keys1 = set(set1.keys())
        keys2 = set(set2.keys())
        
        result = {
            'new': [set2[k] for k in keys2 - keys1],  # 新发现
            'resolved': [set1[k] for k in keys1 - keys2],  # 已消失
            'persisting': [],  # 持续存在
            'changed': []  # 变化（如置信度改变）
        }
        
        # 检查持续存在和变化的诊断
        for key in keys1 & keys2:
            d1 = set1[key]
            d2 = set2[key]
            
            # 检查置信度或严重程度是否变化
            if (d1.get('confidence') != d2.get('confidence') or 
                d1.get('severity') != d2.get('severity')):
                result['changed'].append({
                    'organ': d2.get('organ'),
                    'finding': d2.get('finding'),
                    'before': {
                        'confidence': d1.get('confidence'),
                        'severity': d1.get('severity')
                    },
                    'after': {
                        'confidence': d2.get('confidence'),
                        'severity': d2.get('severity')
                    }
                })
            else:
                result['persisting'].append(d2)
        
        return result
    
    def create_side_by_side_view(self,
                                 image1: np.ndarray,
                                 image2: np.ndarray,
                                 title1: str = "时间点1",
                                 title2: str = "时间点2",
                                 figsize: Tuple = (12, 5)) -> plt.Figure:
        """创建并排对比视图
        
        Args:
            image1: 第一个影像切片
            image2: 第二个影像切片
            title1: 第一个标题
            title2: 第二个标题
            figsize: 图像尺寸
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].imshow(image1, cmap='gray', origin='lower')
        axes[0].set_title(title1)
        axes[0].axis('off')
        
        axes[1].imshow(image2, cmap='gray', origin='lower')
        axes[1].set_title(title2)
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_difference_map(self,
                             image1: np.ndarray,
                             image2: np.ndarray,
                             normalize: bool = True) -> np.ndarray:
        """创建差异图
        
        Args:
            image1: 第一个影像
            image2: 第二个影像
            normalize: 是否归一化
            
        Returns:
            np.ndarray: 差异图
        """
        # 确保形状一致
        if image1.shape != image2.shape:
            raise ValueError(f"影像尺寸不一致: {image1.shape} vs {image2.shape}")
        
        # 计算差异
        diff = image2.astype(float) - image1.astype(float)
        
        if normalize:
            # 归一化到[-1, 1]
            max_val = max(abs(diff.min()), abs(diff.max()))
            if max_val > 0:
                diff = diff / max_val
        
        return diff
    
    def create_comparison_report(self, comparison: Dict) -> str:
        """生成对比报告文本
        
        Args:
            comparison: 对比结果字典
            
        Returns:
            str: 报告文本
        """
        report = "# 影像对比报告\n\n"
        
        report += f"**检查时间**: {comparison['study1_date']} vs {comparison['study2_date']}\n"
        if comparison['time_interval_days']:
            report += f"**时间间隔**: {comparison['time_interval_days']} 天\n\n"
        
        report += "## 诊断变化\n\n"
        
        changes = comparison['diagnosis_changes']
        
        # 新发现
        if changes['new']:
            report += f"### 🆕 新发现 ({len(changes['new'])})\n"
            for diag in changes['new']:
                report += f"- **{diag['organ']}**: {diag['finding']} "
                report += f"(置信度: {diag.get('confidence', 0):.1%}, 严重程度: {diag.get('severity')})\n"
            report += "\n"
        
        # 已消失
        if changes['resolved']:
            report += f"### ✅ 已消失 ({len(changes['resolved'])})\n"
            for diag in changes['resolved']:
                report += f"- **{diag['organ']}**: {diag['finding']}\n"
            report += "\n"
        
        # 变化
        if changes['changed']:
            report += f"### 🔄 变化 ({len(changes['changed'])})\n"
            for change in changes['changed']:
                report += f"- **{change['organ']}**: {change['finding']}\n"
                report += f"  - 置信度: {change['before']['confidence']:.1%} → {change['after']['confidence']:.1%}\n"
                report += f"  - 严重程度: {change['before']['severity']} → {change['after']['severity']}\n"
            report += "\n"
        
        # 持续存在
        if changes['persisting']:
            report += f"### ⏳ 持续存在 ({len(changes['persisting'])})\n"
            for diag in changes['persisting']:
                report += f"- **{diag['organ']}**: {diag['finding']}\n"
            report += "\n"
        
        # 总结
        report += "## 总结\n\n"
        if changes['new']:
            report += f"⚠️ 发现 {len(changes['new'])} 处新异常，建议关注。\n"
        if changes['resolved']:
            report += f"✅ {len(changes['resolved'])} 处异常已消失，情况好转。\n"
        if changes['changed']:
            report += f"📊 {len(changes['changed'])} 处异常有变化，建议进一步评估。\n"
        if not (changes['new'] or changes['changed']):
            report += "✅ 整体情况稳定，无明显新发现或变化。\n"
        
        return report
    
    def calculate_similarity_score(self,
                                   image1: np.ndarray,
                                   image2: np.ndarray,
                                   method: str = 'mse') -> float:
        """计算影像相似度评分
        
        Args:
            image1: 第一个影像
            image2: 第二个影像
            method: 方法 ('mse', 'ssim', 'correlation')
            
        Returns:
            float: 相似度评分
        """
        if image1.shape != image2.shape:
            raise ValueError("影像尺寸不一致")
        
        if method == 'mse':
            # 均方误差（越小越相似）
            mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
            # 转换为相似度（0-1，1表示完全相同）
            max_val = max(image1.max(), image2.max())
            if max_val > 0:
                similarity = 1 - (np.sqrt(mse) / max_val)
            else:
                similarity = 1.0
            return max(0, min(1, similarity))
        
        elif method == 'correlation':
            # 相关系数
            flat1 = image1.flatten().astype(float)
            flat2 = image2.flatten().astype(float)
            correlation = np.corrcoef(flat1, flat2)[0, 1]
            return max(0, min(1, (correlation + 1) / 2))  # 归一化到[0, 1]
        
        else:
            raise ValueError(f"不支持的方法: {method}")


# 测试代码
if __name__ == "__main__":
    comparator = ImageComparator()
    
    # 测试数据
    test_study1 = {
        'date': '2026-01-01',
        'diagnoses': [
            {'organ': '肝', 'finding': '肝囊肿', 'confidence': 0.92, 'severity': 'low'}
        ]
    }
    
    test_study2 = {
        'date': '2026-06-01',
        'diagnoses': [
            {'organ': '肝', 'finding': '肝囊肿', 'confidence': 0.95, 'severity': 'low'},
            {'organ': '肾', 'finding': '肾结石', 'confidence': 0.88, 'severity': 'medium'}
        ]
    }
    
    comparison = comparator.compare_two_studies(test_study1, test_study2)
    print("对比结果:", comparison)
    
    report = comparator.create_comparison_report(comparison)
    print("\n对比报告:")
    print(report)
