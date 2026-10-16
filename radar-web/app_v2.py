"""
RADAR医学影像诊断Web平台 - Streamlit主应用 (Phase 2增强版)

新增功能:
- DICOM系列完整支持和元数据显示
- 3D体绘制可视化
- 器官分割mask叠加
- 病灶自动定位和高亮
"""
import os
import sys
import streamlit as st
import tempfile
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from modules.file_handler import FileHandler
from modules.image_viewer import ImageViewer, load_image_for_viewer
from modules.inference_engine import MockInferenceEngine, RADARInferenceEngine
from modules.visualization import DiagnosisVisualizer
from modules.viewer_3d import Viewer3D, create_simple_3d_view
from utils.dicom_utils import DICOMHandler
from utils.lesion_localizer import LesionLocalizer


# 页面配置
st.set_page_config(
    page_title="RADAR医学影像诊断平台 v2.0",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .feature-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# 初始化session state
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'inference_result' not in st.session_state:
    st.session_state.inference_result = None
if 'viewer' not in st.session_state:
    st.session_state.viewer = None
if 'dicom_metadata' not in st.session_state:
    st.session_state.dicom_metadata = None
if 'show_mask_overlay' not in st.session_state:
    st.session_state.show_mask_overlay = False
if 'mock_mask' not in st.session_state:
    st.session_state.mock_mask = None


def create_mock_segmentation(image_shape):
    """创建模拟的分割mask（用于演示）"""
    mask = np.zeros(image_shape, dtype=np.uint8)
    # 模拟几个器官区域
    z, y, x = image_shape
    mask[z//3:z//3+z//6, y//3:y//3+y//4, x//3:x//3+x//4] = 1  # 器官1
    mask[z//2:z//2+z//8, y//2:y//2+y//6, x//2:x//2+x//6] = 2  # 器官2
    return mask


def main():
    """主应用入口"""
    
    # 标题
    st.markdown('<div class="main-header">🏥 RADAR医学影像诊断平台 <span class="feature-badge">v2.0</span></div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">阿里巴巴达摩院 - 腹部CT智能诊断系统 | Phase 2增强版</div>', 
                unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.title("📂 控制面板")
        
        # 版本信息
        with st.expander("🆕 v2.0 新功能"):
            st.write("✨ DICOM系列完整支持")
            st.write("✨ 3D体绘制可视化")
            st.write("✨ 器官分割叠加显示")
            st.write("✨ 病灶自动定位")
        
        st.markdown("---")
        
        # 文件上传
        st.subheader("1. 上传影像")
        uploaded_file = st.file_uploader(
            "选择CT影像文件",
            type=['nii', 'gz', 'dcm', 'mhd', 'nrrd'],
            help="支持NIfTI (.nii, .nii.gz), DICOM (.dcm), MHD (.mhd), NRRD (.nrrd)格式"
        )
        
        # 处理上传文件
        if uploaded_file is not None:
            if st.session_state.uploaded_file_path is None or \
               not os.path.exists(st.session_state.uploaded_file_path):
                
                with st.spinner("正在处理文件..."):
                    # 保存上传文件
                    temp_dir = tempfile.mkdtemp()
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 转换为NIfTI格式
                    handler = FileHandler()
                    format_type = handler.detect_format(file_path)
                    
                    # 处理DICOM
                    if format_type == 'dicom':
                        dicom_handler = DICOMHandler()
                        if dicom_handler.is_dicom_file(file_path):
                            # 单个DICOM文件或目录
                            st.session_state.dicom_metadata = dicom_handler.extract_metadata(file_path)
                    
                    if format_type == 'nifti':
                        nifti_path = file_path
                    else:
                        nifti_path = handler.convert_to_nifti(file_path)
                    
                    st.session_state.uploaded_file_path = nifti_path
                    
                    # 加载到查看器
                    st.session_state.viewer = load_image_for_viewer(nifti_path)
                    
                    # 创建模拟分割mask
                    st.session_state.mock_mask = create_mock_segmentation(
                        st.session_state.viewer.image_array.shape
                    )
                    
                    st.success(f"✓ 文件已加载 ({format_type})")
        
        st.markdown("---")
        
        # 影像查看控制
        if st.session_state.viewer is not None:
            st.subheader("2. 影像查看")
            
            # 窗宽窗位预设
            window_preset = st.selectbox(
                "窗宽窗位预设",
                options=list(ImageViewer.WINDOW_PRESETS.keys()),
                index=0
            )
            st.session_state.viewer.set_window_preset(window_preset)
            
            # Mask叠加控制
            st.session_state.show_mask_overlay = st.checkbox(
                "显示器官分割叠加",
                value=st.session_state.show_mask_overlay,
                help="在影像上叠加器官分割结果"
            )
            
            if st.session_state.show_mask_overlay:
                mask_opacity = st.slider("叠加透明度", 0.0, 1.0, 0.3, 0.1)
            
            st.markdown("---")
        
        # 推理控制
        if st.session_state.uploaded_file_path is not None:
            st.subheader("3. 开始诊断")
            
            use_mock = st.checkbox("使用模拟推理（开发模式）", value=True)
            
            if st.button("🔍 开始诊断", type="primary", use_container_width=True):
                with st.spinner("正在分析影像，请稍候..."):
                    # 选择推理引擎
                    if use_mock:
                        engine = MockInferenceEngine()
                    else:
                        engine = RADARInferenceEngine(
                            model_root="../ckpt",
                            configs_root="../ckpt"
                        )
                    
                    # 执行推理
                    result = engine.infer(st.session_state.uploaded_file_path)
                    st.session_state.inference_result = result
                    
                    st.success("✓ 诊断完成！")
            
            # 清除按钮
            if st.button("🗑️ 清除数据", use_container_width=True):
                st.session_state.uploaded_file_path = None
                st.session_state.inference_result = None
                st.session_state.viewer = None
                st.session_state.dicom_metadata = None
                st.session_state.mock_mask = None
                st.rerun()
    
    # 主界面
    if st.session_state.uploaded_file_path is None:
        # 欢迎页面
        st.info("👈 请从左侧上传CT影像文件开始诊断")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("### 📁 多格式支持")
            st.write("NIfTI, DICOM, MHD, NRRD")
        with col2:
            st.markdown("### 🔍 智能诊断")
            st.write("36器官 + 200+诊断项")
        with col3:
            st.markdown("### 📊 可视化")
            st.write("2D/3D多视图展示")
        with col4:
            st.markdown("### 🎯 病灶定位")
            st.write("自动定位异常区域")
        
    else:
        # 创建标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 影像查看", 
            "🧊 3D可视化", 
            "📋 诊断报告", 
            "📈 统计分析",
            "ℹ️ 影像信息"
        ])
        
        # Tab 1: 影像查看
        with tab1:
            if st.session_state.viewer is not None:
                st.subheader("多平面重建视图 (MPR)")
                
                viewer = st.session_state.viewer
                
                # 切片位置控制
                col1, col2, col3 = st.columns(3)
                with col1:
                    axial_min, axial_max = viewer.get_slice_range('axial')
                    axial_idx = st.slider(
                        "轴位切片",
                        min_value=int(axial_min),
                        max_value=int(axial_max),
                        value=viewer.axial_idx,
                        key="axial_slider"
                    )
                    viewer.axial_idx = axial_idx
                
                with col2:
                    sagittal_min, sagittal_max = viewer.get_slice_range('sagittal')
                    sagittal_idx = st.slider(
                        "矢状位切片",
                        min_value=int(sagittal_min),
                        max_value=int(sagittal_max),
                        value=viewer.sagittal_idx,
                        key="sagittal_slider"
                    )
                    viewer.sagittal_idx = sagittal_idx
                
                with col3:
                    coronal_min, coronal_max = viewer.get_slice_range('coronal')
                    coronal_idx = st.slider(
                        "冠状位切片",
                        min_value=int(coronal_min),
                        max_value=int(coronal_max),
                        value=viewer.coronal_idx,
                        key="coronal_slider"
                    )
                    viewer.coronal_idx = coronal_idx
                
                # 显示三视图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # 准备mask（如果需要叠加）
                mask_axial = None
                mask_sagittal = None
                mask_coronal = None
                
                if st.session_state.show_mask_overlay and st.session_state.mock_mask is not None:
                    mock_mask = st.session_state.mock_mask
                    mask_axial = mock_mask[:, :, axial_idx]
                    mask_sagittal = mock_mask[sagittal_idx, :, :]
                    mask_coronal = mock_mask[:, coronal_idx, :]
                    
                    # 定义器官颜色
                    label_colors = {1: '#FF6B6B', 2: '#4ECDC4'}
                
                # 轴位
                axial = viewer.get_axial_slice(axial_idx)
                if mask_axial is not None:
                    axial = viewer.overlay_mask(axial, mask_axial, alpha=0.3, label_colors=label_colors)
                axes[0].imshow(axial, cmap='gray' if mask_axial is None else None, origin='lower')
                axes[0].set_title(f'轴位 (Axial) - 切片 {axial_idx}/{viewer.shape[2]}')
                axes[0].axis('off')
                
                # 矢状位
                sagittal = viewer.get_sagittal_slice(sagittal_idx)
                if mask_sagittal is not None:
                    sagittal = viewer.overlay_mask(sagittal, mask_sagittal, alpha=0.3, label_colors=label_colors)
                axes[1].imshow(sagittal, cmap='gray' if mask_sagittal is None else None, origin='lower')
                axes[1].set_title(f'矢状位 (Sagittal) - 切片 {sagittal_idx}/{viewer.shape[0]}')
                axes[1].axis('off')
                
                # 冠状位
                coronal = viewer.get_coronal_slice(coronal_idx)
                if mask_coronal is not None:
                    coronal = viewer.overlay_mask(coronal, mask_coronal, alpha=0.3, label_colors=label_colors)
                axes[2].imshow(coronal, cmap='gray' if mask_coronal is None else None, origin='lower')
                axes[2].set_title(f'冠状位 (Coronal) - 切片 {coronal_idx}/{viewer.shape[1]}')
                axes[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        
        # Tab 2: 3D可视化
        with tab2:
            st.subheader("3D体绘制可视化")
            
            if st.session_state.mock_mask is not None:
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    st.write("**可视化选项**")
                    viz_type = st.radio(
                        "选择可视化类型",
                        ["器官1", "器官2", "全部器官"],
                        help="选择要显示的器官"
                    )
                    
                    opacity_3d = st.slider("3D透明度", 0.1, 1.0, 0.6, 0.1)
                    step_size = st.slider("网格精度", 1, 4, 2, help="数值越小越精细，但渲染越慢")
                
                with col1:
                    with st.spinner("正在生成3D视图..."):
                        if viz_type == "器官1":
                            mask_label = 1
                            fig_3d = create_simple_3d_view(
                                st.session_state.mock_mask,
                                "器官1",
                                '#FF6B6B'
                            )
                        elif viz_type == "器官2":
                            mask_label = 2
                            fig_3d = create_simple_3d_view(
                                st.session_state.mock_mask,
                                "器官2",
                                '#4ECDC4'
                            )
                        else:
                            viewer_3d = Viewer3D()
                            organ_labels = {1: '器官1', 2: '器官2'}
                            fig_3d = viewer_3d.create_multi_organ_visualization(
                                st.session_state.mock_mask,
                                organ_labels,
                                step_size=step_size,
                                opacity=opacity_3d
                            )
                        
                        st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("3D可视化需要分割结果。请先执行诊断。")
        
        # Tab 3: 诊断报告
        with tab3:
            if st.session_state.inference_result is not None:
                result = st.session_state.inference_result
                visualizer = DiagnosisVisualizer()
                
                # 生成报告
                report_md = visualizer.format_diagnosis_report(result)
                st.markdown(report_md)
                
                # 下载按钮
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 下载JSON报告",
                        data=json.dumps(result, ensure_ascii=False, indent=2),
                        file_name=f"report_{result['patient_id']}.json",
                        mime="application/json"
                    )
                with col2:
                    st.download_button(
                        label="📥 下载文本报告",
                        data=report_md,
                        file_name=f"report_{result['patient_id']}.md",
                        mime="text/markdown"
                    )
            else:
                st.info("请先执行诊断以查看报告")
        
        # Tab 4: 统计分析
        with tab4:
            if st.session_state.inference_result is not None:
                result = st.session_state.inference_result
                visualizer = DiagnosisVisualizer()
                
                # 器官检测摘要
                st.subheader("器官检测摘要")
                organ_summary = visualizer.create_organ_summary_table(
                    result.get('organs_detected', [])
                )
                st.dataframe(organ_summary, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # 诊断结果表格
                if result.get('diagnoses'):
                    st.subheader("诊断结果详表")
                    diag_table = visualizer.create_diagnosis_table(result['diagnoses'])
                    st.dataframe(diag_table, use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    
                    # 可视化图表
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_severity = visualizer.create_severity_pie_chart(result['diagnoses'])
                        st.plotly_chart(fig_severity, use_container_width=True)
                    
                    with col2:
                        fig_confidence = visualizer.create_confidence_histogram(result['diagnoses'])
                        st.plotly_chart(fig_confidence, use_container_width=True)
                    
                    # 器官系统统计
                    fig_system = visualizer.create_organ_system_bar_chart(result['diagnoses'])
                    st.plotly_chart(fig_system, use_container_width=True)
                else:
                    st.success("✅ 未发现明显异常")
            else:
                st.info("请先执行诊断以查看统计分析")
        
        # Tab 5: 影像信息
        with tab5:
            st.subheader("影像详细信息")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📐 基本信息")
                if st.session_state.viewer:
                    viewer = st.session_state.viewer
                    st.write(f"**影像尺寸**: {viewer.shape[0]} × {viewer.shape[1]} × {viewer.shape[2]}")
                    st.write(f"**体素间距**: {viewer.spacing[0]:.2f} × {viewer.spacing[1]:.2f} × {viewer.spacing[2]:.2f} mm")
                    st.write(f"**文件名**: {Path(st.session_state.uploaded_file_path).name}")
                    
                    # 计算影像范围
                    volume_mm3 = viewer.shape[0] * viewer.shape[1] * viewer.shape[2] * \
                                viewer.spacing[0] * viewer.spacing[1] * viewer.spacing[2]
                    st.write(f"**扫描体积**: {volume_mm3/1000:.1f} cm³")
            
            with col2:
                st.markdown("### 🏥 DICOM元数据")
                if st.session_state.dicom_metadata:
                    dicom_handler = DICOMHandler()
                    metadata_text = dicom_handler.format_metadata_display(
                        st.session_state.dicom_metadata,
                        anonymize=True
                    )
                    st.text(metadata_text)
                else:
                    st.info("当前文件非DICOM格式或未包含元数据")


if __name__ == "__main__":
    main()
