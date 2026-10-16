"""
RADAR医学影像诊断Web平台 v3.0 - 完整版

集成功能:
Phase 1: 文件上传、2D查看、模拟推理、基础报告
Phase 2: DICOM支持、3D可视化、mask叠加、病灶定位
Phase 3: PDF报告、批量处理、历史记录、影像对比
"""
import os
import sys
import streamlit as st
import tempfile
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from modules.file_handler import FileHandler
from modules.image_viewer import ImageViewer, load_image_for_viewer
from modules.inference_engine import MockInferenceEngine, RADARInferenceEngine
from modules.visualization import DiagnosisVisualizer
from modules.viewer_3d import Viewer3D, create_simple_3d_view
from modules.pdf_generator import PDFReportGenerator
from modules.batch_processor import BatchProcessor, BatchUploadManager
from utils.dicom_utils import DICOMHandler
from utils.lesion_localizer import LesionLocalizer
from utils.history_manager import HistoryManager
from utils.image_comparator import ImageComparator


# 页面配置
st.set_page_config(
    page_title="RADAR医学影像诊断平台 v3.0",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .version-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.9rem;
        margin-left: 1rem;
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# 初始化session state
def init_session_state():
    defaults = {
        'uploaded_file_path': None,
        'inference_result': None,
        'viewer': None,
        'dicom_metadata': None,
        'show_mask_overlay': False,
        'mock_mask': None,
        'history_manager': HistoryManager("radar_history.db"),
        'batch_results': [],
        'comparison_studies': [],
        'current_case_id': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


def create_mock_segmentation(image_shape):
    """创建模拟的分割mask"""
    mask = np.zeros(image_shape, dtype=np.uint8)
    z, y, x = image_shape
    mask[z//3:z//3+z//6, y//3:y//3+y//4, x//3:x//3+x//4] = 1
    mask[z//2:z//2+z//8, y//2:y//2+y//6, x//2:x//2+x//6] = 2
    return mask


def main():
    """主应用入口"""
    
    # 标题
    st.markdown(
        '<div class="main-header">🏥 RADAR医学影像诊断平台 '
        '<span class="version-badge">v3.0 全功能版</span></div>', 
        unsafe_allow_html=True
    )
    
    # 侧边栏
    with st.sidebar:
        st.title("📂 控制面板")
        
        # 版本信息
        with st.expander("🆕 v3.0 完整功能"):
            st.markdown("""
            **Phase 1**: 基础功能  
            **Phase 2**: 3D可视化  
            **Phase 3**: 
            - ✨ PDF专业报告
            - ✨ 批量处理
            - ✨ 历史记录
            - ✨ 影像对比
            """)
        
        st.markdown("---")
        
        # 模式选择
        mode = st.radio(
            "选择工作模式",
            ["单文件诊断", "批量处理", "历史记录", "影像对比"],
            help="选择不同的工作模式"
        )
        
        st.markdown("---")
        
        # 根据模式显示不同控制
        if mode == "单文件诊断":
            render_single_file_controls()
        elif mode == "批量处理":
            render_batch_controls()
        elif mode == "历史记录":
            render_history_controls()
        elif mode == "影像对比":
            render_comparison_controls()
    
    # 主界面 - 根据模式显示不同内容
    if mode == "单文件诊断":
        render_single_file_interface()
    elif mode == "批量处理":
        render_batch_interface()
    elif mode == "历史记录":
        render_history_interface()
    elif mode == "影像对比":
        render_comparison_interface()


def render_single_file_controls():
    """单文件模式控制"""
    st.subheader("1. 上传影像")
    uploaded_file = st.file_uploader(
        "选择CT影像文件",
        type=['nii', 'gz', 'dcm', 'mhd', 'nrrd'],
        key="single_file_uploader"
    )
    
    if uploaded_file is not None:
        if st.session_state.uploaded_file_path is None or \
           not os.path.exists(st.session_state.uploaded_file_path):
            
            with st.spinner("正在处理文件..."):
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                handler = FileHandler()
                format_type = handler.detect_format(file_path)
                
                if format_type == 'dicom':
                    dicom_handler = DICOMHandler()
                    if dicom_handler.is_dicom_file(file_path):
                        st.session_state.dicom_metadata = dicom_handler.extract_metadata(file_path)
                
                nifti_path = file_path if format_type == 'nifti' else handler.convert_to_nifti(file_path)
                st.session_state.uploaded_file_path = nifti_path
                st.session_state.viewer = load_image_for_viewer(nifti_path)
                st.session_state.mock_mask = create_mock_segmentation(
                    st.session_state.viewer.image_array.shape
                )
                
                st.success(f"✓ 文件已加载 ({format_type})")
    
    st.markdown("---")
    
    if st.session_state.viewer is not None:
        st.subheader("2. 查看选项")
        
        window_preset = st.selectbox(
            "窗宽窗位预设",
            options=list(ImageViewer.WINDOW_PRESETS.keys()),
            index=0,
            key="window_preset"
        )
        st.session_state.viewer.set_window_preset(window_preset)
        
        st.session_state.show_mask_overlay = st.checkbox(
            "显示器官分割",
            value=st.session_state.show_mask_overlay
        )
        
        st.markdown("---")
    
    if st.session_state.uploaded_file_path is not None:
        st.subheader("3. 开始诊断")
        
        use_mock = st.checkbox("使用模拟推理", value=True, key="use_mock")
        save_to_history = st.checkbox("保存到历史记录", value=True)
        
        if st.button("🔍 开始诊断", type="primary", use_container_width=True):
            with st.spinner("正在分析影像..."):
                engine = MockInferenceEngine() if use_mock else RADARInferenceEngine()
                result = engine.infer(st.session_state.uploaded_file_path)
                st.session_state.inference_result = result
                
                # 保存到历史记录
                if save_to_history:
                    case_id = st.session_state.history_manager.add_case(
                        patient_id=result['patient_id'],
                        file_name=Path(st.session_state.uploaded_file_path).name,
                        file_path=st.session_state.uploaded_file_path,
                        diagnosis_result=result
                    )
                    st.session_state.current_case_id = case_id
                
                st.success("✓ 诊断完成！")
        
        if st.button("🗑️ 清除数据", use_container_width=True):
            for key in ['uploaded_file_path', 'inference_result', 'viewer', 'mock_mask', 'current_case_id']:
                st.session_state[key] = None
            st.rerun()


def render_batch_controls():
    """批量处理模式控制"""
    st.subheader("批量文件上传")
    
    uploaded_files = st.file_uploader(
        "选择多个CT影像文件",
        type=['nii', 'gz'],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    if uploaded_files:
        st.write(f"已选择 {len(uploaded_files)} 个文件")
        
        use_mock = st.checkbox("使用模拟推理", value=True, key="batch_mock")
        max_workers = st.slider("并行数", 1, 5, 2)
        
        if st.button("🚀 开始批量处理", type="primary", use_container_width=True):
            with st.spinner(f"正在处理 {len(uploaded_files)} 个文件..."):
                # 保存上传文件
                upload_manager = BatchUploadManager()
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = upload_manager.add_file(
                        uploaded_file.getbuffer(),
                        uploaded_file.name
                    )
                    file_paths.append(file_path)
                
                # 批量处理
                engine = MockInferenceEngine() if use_mock else RADARInferenceEngine()
                processor = BatchProcessor(engine, max_workers=max_workers)
                
                progress_bar = st.progress(0)
                def progress_callback(current, total, file_path):
                    progress_bar.progress(current / total)
                
                results = processor.process_file_list(file_paths, progress_callback)
                st.session_state.batch_results = results
                
                # 保存到历史记录
                for result in results:
                    if result.get('status') != 'failed':
                        st.session_state.history_manager.add_case(
                            patient_id=result['patient_id'],
                            file_name=Path(result['file_path']).name,
                            diagnosis_result=result
                        )
                
                st.success(f"✓ 完成！成功处理 {len([r for r in results if r.get('status') != 'failed'])} 个文件")


def render_history_controls():
    """历史记录模式控制"""
    st.subheader("历史记录搜索")
    
    search_keyword = st.text_input("搜索关键词（患者ID或文件名）")
    
    col1, col2 = st.columns(2)
    with col1:
        severity_filter = st.selectbox(
            "严重程度筛选",
            ["全部", "high", "medium", "low"]
        )
    with col2:
        date_range = st.date_input(
            "日期范围",
            value=[]
        )
    
    if st.button("🔍 搜索", use_container_width=True):
        st.session_state['search_triggered'] = True


def render_comparison_controls():
    """影像对比模式控制"""
    st.subheader("选择对比影像")
    
    st.write("**第一次检查**")
    study1_file = st.file_uploader("上传第一次检查", type=['nii', 'gz'], key="study1")
    study1_date = st.date_input("检查日期", key="study1_date")
    
    st.markdown("---")
    
    st.write("**第二次检查**")
    study2_file = st.file_uploader("上传第二次检查", type=['nii', 'gz'], key="study2")
    study2_date = st.date_input("检查日期", key="study2_date")
    
    if st.button("🔄 开始对比", type="primary", use_container_width=True):
        if study1_file and study2_file:
            st.session_state['comparison_ready'] = True


def render_single_file_interface():
    """单文件模式主界面"""
    if st.session_state.uploaded_file_path is None:
        st.info("👈 请从左侧上传CT影像文件开始诊断")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("### 📁 多格式")
            st.write("NIfTI, DICOM等")
        with col2:
            st.markdown("### 🔍 智能诊断")
            st.write("36器官+200+项")
        with col3:
            st.markdown("### 📊 多维可视化")
            st.write("2D/3D全方位")
        with col4:
            st.markdown("### 📄 PDF报告")
            st.write("专业诊断报告")
    else:
        tabs = st.tabs([
            "📊 影像查看",
            "🧊 3D可视化",
            "📋 诊断报告",
            "📈 统计分析",
            "ℹ️ 影像信息"
        ])
        
        with tabs[0]:
            render_image_view()
        
        with tabs[1]:
            render_3d_view()
        
        with tabs[2]:
            render_diagnosis_report()
        
        with tabs[3]:
            render_statistics()
        
        with tabs[4]:
            render_image_info()


def render_batch_interface():
    """批量处理界面"""
    st.header("批量处理结果")
    
    if not st.session_state.batch_results:
        st.info("请从左侧上传多个文件并开始批量处理")
    else:
        results = st.session_state.batch_results
        
        # 摘要统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总文件数", len(results))
        with col2:
            success = len([r for r in results if r.get('status') != 'failed'])
            st.metric("成功", success)
        with col3:
            failed = len([r for r in results if r.get('status') == 'failed'])
            st.metric("失败", failed)
        with col4:
            total_diag = sum(len(r.get('diagnoses', [])) for r in results)
            st.metric("总诊断数", total_diag)
        
        st.markdown("---")
        
        # 结果列表
        st.subheader("处理详情")
        for i, result in enumerate(results, 1):
            with st.expander(f"#{i} - {result.get('patient_id', 'Unknown')}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**文件**: {Path(result['file_path']).name}")
                    st.write(f"**器官数**: {len(result.get('organs_detected', []))}")
                    st.write(f"**诊断数**: {len(result.get('diagnoses', []))}")
                with col2:
                    status = result.get('status', 'success')
                    if status == 'failed':
                        st.error(f"失败: {result.get('error', '')}")
                    else:
                        st.success("成功")
        
        # 导出选项
        st.markdown("---")
        st.subheader("导出结果")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 导出JSON"):
                json_str = json.dumps(results, ensure_ascii=False, indent=2)
                st.download_button(
                    "下载JSON",
                    json_str,
                    "batch_results.json",
                    "application/json"
                )
        with col2:
            if st.button("📥 导出CSV"):
                st.info("CSV导出功能需要pandas处理")


def render_history_interface():
    """历史记录界面"""
    st.header("历史记录管理")
    
    manager = st.session_state.history_manager
    
    # 统计概览
    stats = manager.get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总病例数", stats['total_cases'])
    with col2:
        st.metric("总诊断数", stats['total_diagnoses'])
    with col3:
        high_count = stats['severity_distribution'].get('high', 0)
        st.metric("高危发现", high_count)
    with col4:
        recent = len(stats.get('recent_activity', []))
        st.metric("近7天", recent)
    
    st.markdown("---")
    
    # 病例列表
    st.subheader("病例列表")
    
    # 搜索功能
    if st.session_state.get('search_triggered'):
        cases = manager.list_cases(limit=50)
    else:
        cases = manager.list_cases(limit=20)
    
    if cases:
        for case in cases:
            with st.expander(f"病例 #{case['id']} - {case['patient_id']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**文件**: {case['file_name']}")
                    st.write(f"**时间**: {case['upload_time']}")
                    st.write(f"**诊断数**: {case.get('diagnoses_count', 0)}")
                with col2:
                    if st.button("查看详情", key=f"view_{case['id']}"):
                        st.session_state.current_case_id = case['id']
                        st.rerun()
    else:
        st.info("暂无历史记录")


def render_comparison_interface():
    """影像对比界面"""
    st.header("多时间点影像对比")
    
    if not st.session_state.get('comparison_ready'):
        st.info("👈 请从左侧上传两次检查的影像文件")
    else:
        st.success("对比分析准备就绪")
        st.info("影像对比功能展示（需要实际数据集成）")


def render_image_view():
    """影像查看标签页"""
    if st.session_state.viewer is not None:
        st.subheader("多平面重建视图 (MPR)")
        
        viewer = st.session_state.viewer
        
        col1, col2, col3 = st.columns(3)
        with col1:
            axial_min, axial_max = viewer.get_slice_range('axial')
            axial_idx = st.slider("轴位", int(axial_min), int(axial_max), viewer.axial_idx, key="axial")
            viewer.axial_idx = axial_idx
        with col2:
            sagittal_min, sagittal_max = viewer.get_slice_range('sagittal')
            sagittal_idx = st.slider("矢状位", int(sagittal_min), int(sagittal_max), viewer.sagittal_idx, key="sagittal")
            viewer.sagittal_idx = sagittal_idx
        with col3:
            coronal_min, coronal_max = viewer.get_slice_range('coronal')
            coronal_idx = st.slider("冠状位", int(coronal_min), int(coronal_max), viewer.coronal_idx, key="coronal")
            viewer.coronal_idx = coronal_idx
        
        # 显示三视图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        mask_axial = mask_sagittal = mask_coronal = None
        label_colors = {1: '#FF6B6B', 2: '#4ECDC4'}
        
        if st.session_state.show_mask_overlay and st.session_state.mock_mask is not None:
            mock_mask = st.session_state.mock_mask
            mask_axial = mock_mask[:, :, axial_idx]
            mask_sagittal = mock_mask[sagittal_idx, :, :]
            mask_coronal = mock_mask[:, coronal_idx, :]
        
        # 三个视图
        for idx, (view_type, mask) in enumerate([
            ('axial', mask_axial),
            ('sagittal', mask_sagittal),
            ('coronal', mask_coronal)
        ]):
            if view_type == 'axial':
                img = viewer.get_axial_slice(axial_idx)
            elif view_type == 'sagittal':
                img = viewer.get_sagittal_slice(sagittal_idx)
            else:
                img = viewer.get_coronal_slice(coronal_idx)
            
            if mask is not None:
                img = viewer.overlay_mask(img, mask, alpha=0.3, label_colors=label_colors)
            
            axes[idx].imshow(img, cmap='gray' if mask is None else None, origin='lower')
            axes[idx].set_title(f'{["轴位", "矢状位", "冠状位"][idx]}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_3d_view():
    """3D可视化标签页"""
    st.subheader("3D体绘制可视化")
    
    if st.session_state.mock_mask is not None:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            viz_type = st.radio("可视化类型", ["器官1", "器官2", "全部器官"])
            opacity_3d = st.slider("透明度", 0.1, 1.0, 0.6, 0.1)
        
        with col1:
            with st.spinner("生成3D视图..."):
                if viz_type == "全部器官":
                    viewer_3d = Viewer3D()
                    organ_labels = {1: '器官1', 2: '器官2'}
                    fig_3d = viewer_3d.create_multi_organ_visualization(
                        st.session_state.mock_mask,
                        organ_labels,
                        step_size=2,
                        opacity=opacity_3d
                    )
                else:
                    label = 1 if viz_type == "器官1" else 2
                    fig_3d = create_simple_3d_view(
                        st.session_state.mock_mask,
                        viz_type,
                        '#FF6B6B' if label == 1 else '#4ECDC4'
                    )
                
                st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("3D可视化需要分割结果")


def render_diagnosis_report():
    """诊断报告标签页"""
    if st.session_state.inference_result is not None:
        result = st.session_state.inference_result
        visualizer = DiagnosisVisualizer()
        
        # Markdown报告
        report_md = visualizer.format_diagnosis_report(result)
        st.markdown(report_md)
        
        st.markdown("---")
        
        # 下载选项
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 下载JSON",
                json.dumps(result, ensure_ascii=False, indent=2),
                f"report_{result['patient_id']}.json",
                "application/json"
            )
        
        with col2:
            st.download_button(
                "📥 下载Markdown",
                report_md,
                f"report_{result['patient_id']}.md",
                "text/markdown"
            )
        
        with col3:
            if st.button("📄 生成PDF报告"):
                with st.spinner("正在生成PDF..."):
                    try:
                        generator = PDFReportGenerator()
                        
                        # 准备影像切片
                        viewer = st.session_state.viewer
                        image_slices = {
                            'axial': viewer.get_axial_slice(),
                            'sagittal': viewer.get_sagittal_slice(),
                            'coronal': viewer.get_coronal_slice()
                        }
                        
                        pdf_path = f"report_{result['patient_id']}.pdf"
                        generator.generate_report(
                            pdf_path,
                            result,
                            image_slices
                        )
                        
                        # 读取PDF并提供下载
                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                "📥 下载PDF",
                                f.read(),
                                pdf_path,
                                "application/pdf"
                            )
                        
                        st.success("✓ PDF报告已生成")
                    except Exception as e:
                        st.error(f"PDF生成失败: {e}")
    else:
        st.info("请先执行诊断")


def render_statistics():
    """统计分析标签页"""
    if st.session_state.inference_result is not None:
        result = st.session_state.inference_result
        visualizer = DiagnosisVisualizer()
        
        # 器官摘要
        st.subheader("器官检测摘要")
        organ_summary = visualizer.create_organ_summary_table(result.get('organs_detected', []))
        st.dataframe(organ_summary, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        if result.get('diagnoses'):
            st.subheader("诊断结果详表")
            diag_table = visualizer.create_diagnosis_table(result['diagnoses'])
            st.dataframe(diag_table, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # 图表
            col1, col2 = st.columns(2)
            with col1:
                fig = visualizer.create_severity_pie_chart(result['diagnoses'])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = visualizer.create_confidence_histogram(result['diagnoses'])
                st.plotly_chart(fig, use_container_width=True)
            
            fig = visualizer.create_organ_system_bar_chart(result['diagnoses'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ 未发现明显异常")
    else:
        st.info("请先执行诊断")


def render_image_info():
    """影像信息标签页"""
    st.subheader("影像详细信息")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📐 基本信息")
        if st.session_state.viewer:
            viewer = st.session_state.viewer
            st.write(f"**尺寸**: {viewer.shape[0]} × {viewer.shape[1]} × {viewer.shape[2]}")
            st.write(f"**间距**: {viewer.spacing[0]:.2f} × {viewer.spacing[1]:.2f} × {viewer.spacing[2]:.2f} mm")
            st.write(f"**文件**: {Path(st.session_state.uploaded_file_path).name}")
    
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
            st.info("非DICOM格式或无元数据")


if __name__ == "__main__":
    main()
