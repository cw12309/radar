"""
3D可视化模块
使用Plotly实现器官和病灶的3D体绘制
"""
import numpy as np
import plotly.graph_objects as go
from skimage import measure
from typing import List, Dict, Tuple, Optional


class Viewer3D:
    """3D体绘制查看器"""
    
    # 器官颜色映射（RGB）
    ORGAN_COLORS = {
        '肝': '#8B4513',
        '肾': '#DC143C',
        '脾': '#800080',
        '胰腺': '#FF69B4',
        '胆囊': '#00FF00',
        '胃': '#FFD700',
        '大肠': '#FFA500',
        '小肠': '#FF8C00',
        '心脏': '#FF0000',
        '肺': '#87CEEB',
        '主动脉': '#B22222',
        '膀胱': '#4169E1',
        '脑': '#FF1493',
        '骨骼': '#F5F5DC',
    }
    
    def __init__(self):
        self.meshes = []
        self.figure = None
    
    def create_mesh_from_mask(self, 
                             mask_array: np.ndarray, 
                             label_value: int = 1,
                             step_size: int = 1,
                             color: str = '#1f77b4',
                             opacity: float = 0.5,
                             name: str = 'Organ') -> go.Mesh3d:
        """从分割mask创建3D网格
        
        Args:
            mask_array: 3D分割mask数组
            label_value: 要提取的标签值
            step_size: 降采样步长（减少顶点数）
            color: 颜色
            opacity: 透明度
            name: 网格名称
            
        Returns:
            plotly.graph_objects.Mesh3d: 3D网格对象
        """
        # 提取指定标签的mask
        binary_mask = (mask_array == label_value).astype(np.uint8)
        
        # 降采样以提高性能
        if step_size > 1:
            binary_mask = binary_mask[::step_size, ::step_size, ::step_size]
        
        # 使用marching cubes算法生成网格
        try:
            verts, faces, normals, values = measure.marching_cubes(
                binary_mask, 
                level=0.5,
                step_size=1
            )
        except Exception as e:
            print(f"警告: 无法生成网格 - {e}")
            return None
        
        # 缩放回原始尺寸
        if step_size > 1:
            verts = verts * step_size
        
        # 创建Mesh3d对象
        mesh = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=name,
            flatshading=True,
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.2,
                roughness=0.5
            ),
            lightposition=dict(
                x=100,
                y=100,
                z=1000
            )
        )
        
        return mesh
    
    def create_multi_organ_visualization(self,
                                        mask_array: np.ndarray,
                                        organ_labels: Dict[int, str],
                                        step_size: int = 2,
                                        opacity: float = 0.6) -> go.Figure:
        """创建多器官3D可视化
        
        Args:
            mask_array: 3D分割mask数组（多标签）
            organ_labels: {标签值: 器官名称} 字典
            step_size: 降采样步长
            opacity: 透明度
            
        Returns:
            plotly.graph_objects.Figure: 3D图形对象
        """
        meshes = []
        
        for label_value, organ_name in organ_labels.items():
            if label_value == 0:  # 跳过背景
                continue
            
            # 获取器官颜色
            color = self.ORGAN_COLORS.get(organ_name, '#888888')
            
            # 创建网格
            mesh = self.create_mesh_from_mask(
                mask_array,
                label_value=label_value,
                step_size=step_size,
                color=color,
                opacity=opacity,
                name=organ_name
            )
            
            if mesh is not None:
                meshes.append(mesh)
        
        # 创建图形
        fig = go.Figure(data=meshes)
        
        # 设置布局
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            title="3D器官可视化",
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=700
        )
        
        self.figure = fig
        return fig
    
    def create_volume_rendering(self, 
                                image_array: np.ndarray,
                                opacity_scale: float = 0.1,
                                colorscale: str = 'gray') -> go.Figure:
        """创建体积渲染（Volume Rendering）
        
        Args:
            image_array: 3D影像数组
            opacity_scale: 透明度缩放
            colorscale: 颜色映射
            
        Returns:
            plotly.graph_objects.Figure: 3D体积渲染图
        """
        # 降采样以提高性能
        image_downsampled = image_array[::2, ::2, ::2]
        
        # 归一化到[0, 1]
        image_norm = (image_downsampled - image_downsampled.min()) / \
                     (image_downsampled.max() - image_downsampled.min() + 1e-8)
        
        # 创建体积对象
        volume = go.Volume(
            x=np.arange(image_norm.shape[0]),
            y=np.arange(image_norm.shape[1]),
            z=np.arange(image_norm.shape[2]),
            value=image_norm.flatten(),
            isomin=0.1,
            isomax=0.9,
            opacity=opacity_scale,
            surface_count=15,
            colorscale=colorscale
        )
        
        fig = go.Figure(data=volume)
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            ),
            title="3D体积渲染",
            margin=dict(l=0, r=0, t=40, b=0),
            height=700
        )
        
        return fig
    
    def add_bounding_box(self, 
                        fig: go.Figure,
                        bbox: Tuple[int, int, int, int, int, int],
                        color: str = 'red',
                        name: str = 'ROI') -> go.Figure:
        """添加边界框
        
        Args:
            fig: Plotly图形对象
            bbox: 边界框坐标 (x_min, y_min, z_min, x_max, y_max, z_max)
            color: 边界框颜色
            name: 边界框名称
            
        Returns:
            plotly.graph_objects.Figure: 更新后的图形
        """
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        
        # 定义边界框的12条边
        edges = [
            # 底面
            ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
            ([x_min, x_min], [y_min, y_max], [z_min, z_min]),
            ([x_max, x_max], [y_min, y_max], [z_min, z_min]),
            ([x_min, x_max], [y_max, y_max], [z_min, z_min]),
            # 顶面
            ([x_min, x_max], [y_min, y_min], [z_max, z_max]),
            ([x_min, x_min], [y_min, y_max], [z_max, z_max]),
            ([x_max, x_max], [y_min, y_max], [z_max, z_max]),
            ([x_min, x_max], [y_max, y_max], [z_max, z_max]),
            # 竖边
            ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
            ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
            ([x_min, x_min], [y_max, y_max], [z_min, z_max]),
            ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
        ]
        
        for i, (x, y, z) in enumerate(edges):
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=(i == 0),
                name=name if i == 0 else None
            ))
        
        return fig


def create_simple_3d_view(mask_array: np.ndarray, 
                         organ_name: str = 'Organ',
                         color: str = '#1f77b4') -> go.Figure:
    """快速创建单个器官的3D视图
    
    Args:
        mask_array: 二值mask数组
        organ_name: 器官名称
        color: 颜色
        
    Returns:
        plotly.graph_objects.Figure: 3D图形
    """
    viewer = Viewer3D()
    mesh = viewer.create_mesh_from_mask(
        mask_array, 
        label_value=1, 
        step_size=2,
        color=color,
        opacity=0.7,
        name=organ_name
    )
    
    if mesh is None:
        # 返回空图
        fig = go.Figure()
        fig.add_annotation(
            text="无法生成3D视图",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        title=f"{organ_name} 3D视图",
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )
    
    return fig


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    test_mask = np.zeros((64, 64, 64), dtype=np.uint8)
    test_mask[20:40, 20:40, 20:40] = 1  # 立方体
    
    viewer = Viewer3D()
    fig = create_simple_3d_view(test_mask, '测试器官', '#FF6B6B')
    print("3D查看器创建成功")
    print(f"图形对象类型: {type(fig)}")
