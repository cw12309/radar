"""
RADAR医学影像诊断Web平台 - Streamlit主应用

Phase 1 功能:
- 文件上传（NIfTI格式）
- 2D切片查看器（三视图）
- 模型推理
- 诊断结果展示
"""
import os
import sys
import streamlit as st
import tempfile
import json
from pathlib import Path
import matplotlib.pyplot as plt

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from modules.file_handler import FileHandler
from modules.image_viewer import ImageViewer, load_image_for_viewer
from modules.inference_engine import MockInferenceEngine, RADARInferenceEngine
from modules.visualization import DiagnosisVisualizer


# 页面配置
st.set_page_config(
    page_title="RADAR医学影像诊断平台",
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
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .status-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
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


def main():
    """主应用入口"""
    
    # 标题
    st.markdown('<div class="main-header">🏥 RADAR医学影像诊断平台</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">阿里巴巴达摩院 - 腹部CT智能诊断系统</div>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.title("📂 控制面板")
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
                    
                    if format_type == 'nifti':
                        nifti_path = file_path
                    else:
                        nifti_path = handler.convert_to_nifti(file_path)
                    
                    st.session_state.uploaded_file_path = nifti_path
                    
                    # 加载到查看器
                    st.session_state.viewer = load_image_for_viewer(nifti_path)
                    
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
            
            # 自定义窗宽窗位
            with st.expander("自定义窗宽窗位"):
                window_center = st.slider(
                    "窗位中心",
                    min_value=-1000,
                    max_value=1000,
                    value=st.session_state.viewer.window_center,
                    step=10
                )
                window_width = st.slider(
                    "窗宽",
                    min_value=1,
                    max_value=2000,
                    value=st.session_state.viewer.window_width,
                    step=10
                )
                st.session_state.viewer.window_center = window_center
                st.session_state.viewer.window_width = window_width
            
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
                st.rerun()
    
    # 主界面
    if st.session_state.uploaded_file_path is None:
        # 欢迎页面
        st.info("👈 请从左侧上传CT影像文件开始诊断")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📁 多格式支持")
            st.write("支持NIfTI, DICOM, MHD, NRRD等医学影像格式")
        with col2:
            st.markdown("### 🔍 智能诊断")
            st.write("基于RADAR模型，识别36个器官，检测200+诊断项")
        with col3:
            st.markdown("### 📊 可视化展示")
            st.write("三视图查看、窗宽窗位调整、结果可视化")
        
    else:
        # 创建标签页
        tab1, tab2, tab3 = st.tabs(["📊 影像查看", "📋 诊断报告", "📈 统计分析"])
        
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
                fig = viewer.create_three_view_figure(figsize=(18, 6))
                st.pyplot(fig)
                plt.close(fig)
                
                # 影像信息
                with st.expander("📄 影像信息"):
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.write(f"**尺寸**: {viewer.shape[0]} × {viewer.shape[1]} × {viewer.shape[2]}")
                        st.write(f"**体素间距**: {viewer.spacing[0]:.2f} × {viewer.spacing[1]:.2f} × {viewer.spacing[2]:.2f} mm")
                    with info_col2:
                        st.write(f"**文件**: {Path(st.session_state.uploaded_file_path).name}")
        
        # Tab 2: 诊断报告
        with tab2:
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
        
        # Tab 3: 统计分析
        with tab3:
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


if __name__ == "__main__":
    main()
