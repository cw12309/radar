"""
RADAR医学影像诊断Web平台 v4.0 - 专业版

新增功能:
- 专业级医学影像查看器（基于现代Web技术）
- 优化的用户界面和交互体验
- 增强的窗位窗宽调整
- 多视图同步显示
- 更美观的整体设计
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'components'))

from modules.file_handler import FileHandler
from modules.image_viewer import load_image_for_viewer
from modules.inference_engine import MockInferenceEngine
from modules.visualization import DiagnosisVisualizer
from modules.pdf_generator import PDFReportGenerator
from modules.batch_processor import BatchProcessor
from utils.history_manager import HistoryManager
from utils.lesion_localizer import LesionLocalizer
from components.medical_viewer import create_cornerstone_viewer, create_multi_view_viewer


# 页面配置
st.set_page_config(
    page_title="RADAR医学影像诊断平台 v4.0",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 现代化CSS样式
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 0;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .version-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(245, 87, 108, 0.3);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #10b98115 0%, #059669 15 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f59e0b15 0%, #f59e0b15 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
    }
    
    h1, h2, h3 {
        color: #1f2937;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background-color: #f3f4f6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# 初始化session state
def init_session_state():
    defaults = {
        'uploaded_file_path': None,
        'inference_result': None,
        'image_data': None,
        'dicom_metadata': None,
        'history_manager': HistoryManager("radar_history.db"),
        'batch_results': [],
        'current_case_id': None,
        'viewer_mode': 'single'  # single 或 multi
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


def main():
    # 标题区域
    st.markdown('<h1 class="main-header">🏥 RADAR 医学影像诊断平台</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">基于达摩院AI的专业腹部CT诊断系统 <span class="version-badge">v4.0 专业版</span></p>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 📋 功能导航")
        page = st.radio(
            "选择功能模块",
            ["🔍 单例诊断", "📊 批量处理", "📜 历史记录", "⚙️ 系统设置"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📈 系统状态")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">✓</div><div class="metric-label">系统在线</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">AI</div><div class="metric-label">模型就绪</div></div>', unsafe_allow_html=True)
    
    # 主页面路由
    if page == "🔍 单例诊断":
        single_diagnosis_page()
    elif page == "📊 批量处理":
        batch_processing_page()
    elif page == "📜 历史记录":
        history_page()
    else:
        settings_page()


def single_diagnosis_page():
    """单例诊断页面"""
    st.markdown("## 🔍 单例影像诊断")
    
    # 文件上传区域
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 📤 上传医学影像")
    
    uploaded_file = st.file_uploader(
        "支持格式: NIfTI (.nii, .nii.gz), DICOM (.dcm), NumPy (.npy), PNG/JPG",
        type=['nii', 'gz', 'dcm', 'npy', 'png', 'jpg', 'jpeg'],
        help="请上传腹部CT影像文件"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        # 处理上传文件
        with st.spinner("🔄 正在加载影像数据..."):
            temp_path = save_uploaded_file(uploaded_file)
            
            try:
                # 加载影像
                file_handler = FileHandler()
                
                # 根据文件扩展名加载
                if temp_path.endswith('.npy'):
                    image_data = np.load(temp_path)
                    file_type = 'numpy'
                elif temp_path.endswith(('.nii', '.nii.gz')):
                    image_data, _, _ = file_handler.load_nifti(temp_path)
                    file_type = 'nifti'
                elif temp_path.endswith(('.dcm', '.dicom')):
                    # 先转换为NIfTI
                    nifti_path = file_handler.convert_to_nifti(temp_path)
                    image_data, _, _ = file_handler.load_nifti(nifti_path)
                    file_type = 'dicom'
                elif temp_path.endswith(('.png', '.jpg', '.jpeg')):
                    from PIL import Image
                    img = Image.open(temp_path)
                    image_data = np.array(img)
                    file_type = 'image'
                else:
                    raise ValueError(f"不支持的文件格式")
                
                st.session_state['image_data'] = image_data
                st.session_state['uploaded_file_path'] = temp_path
                
                st.markdown('<div class="success-box">✅ 影像加载成功！</div>', unsafe_allow_html=True)
                
                # 显示文件信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("文件格式", file_type.upper())
                with col2:
                    st.metric("影像尺寸", f"{image_data.shape}")
                with col3:
                    st.metric("数据类型", str(image_data.dtype))
                with col4:
                    st.metric("切片数", image_data.shape[0] if len(image_data.shape) == 3 else 1)
                
            except Exception as e:
                st.error(f"❌ 加载失败: {str(e)}")
                return
        
        # 影像查看器选项
        st.markdown("---")
        st.markdown("## 🖼️ 专业影像查看器")
        
        viewer_type = st.radio(
            "选择查看模式",
            ["🔵 单视图高级查看器", "🌐 多视图（三视图）", "📊 传统2D查看器"],
            horizontal=True
        )
        
        if len(image_data.shape) == 3:
            if viewer_type == "🔵 单视图高级查看器":
                st.markdown('<div class="info-box">💡 支持平移、缩放、窗位窗宽调整等专业功能</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    slice_idx = st.slider("选择切片", 0, image_data.shape[0]-1, image_data.shape[0]//2)
                    view_mode = st.selectbox("视图方向", ["axial", "sagittal", "coronal"])
                    window_center = st.slider("窗位", -1000, 1000, 40)
                    window_width = st.slider("窗宽", 1, 2000, 400)
                
                with col2:
                    create_cornerstone_viewer(
                        image_data,
                        window_center=window_center,
                        window_width=window_width,
                        slice_index=slice_idx,
                        view_mode=view_mode
                    )
            
            elif viewer_type == "🌐 多视图（三视图）":
                st.markdown('<div class="info-box">💡 同时显示轴位、矢状位、冠状位三个视图</div>', unsafe_allow_html=True)
                create_multi_view_viewer(image_data)
            
            else:
                st.markdown('<div class="info-box">💡 传统2D切片浏览模式</div>', unsafe_allow_html=True)
                slice_idx = st.slider("选择切片", 0, image_data.shape[0]-1, image_data.shape[0]//2)
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(image_data[slice_idx], cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
        else:
            # 2D图像
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image_data, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
        
        # AI诊断区域
        st.markdown("---")
        st.markdown("## 🤖 AI智能诊断")
        
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🚀 开始AI诊断分析", use_container_width=True):
                with st.spinner("🧠 AI模型正在分析影像..."):
                    run_inference(image_data, temp_path)
        
        # 显示诊断结果
        if st.session_state.get('inference_result'):
            display_diagnosis_results(st.session_state['inference_result'])


def run_inference(image_data, file_path):
    """执行推理"""
    try:
        engine = MockInferenceEngine()
        result = engine.infer(image_data)
        st.session_state['inference_result'] = result
        
        # 保存到历史记录
        case_id = st.session_state['history_manager'].add_record(
            patient_id=f"P{datetime.now().strftime('%Y%m%d%H%M%S')}",
            file_path=file_path,
            diagnosis_result=json.dumps(result, ensure_ascii=False),
            image_shape=str(image_data.shape)
        )
        st.session_state['current_case_id'] = case_id
        
        st.success("✅ 诊断完成！")
        
    except Exception as e:
        st.error(f"❌ 诊断失败: {str(e)}")


def display_diagnosis_results(result):
    """显示诊断结果"""
    st.markdown("---")
    st.markdown("## 📋 诊断报告")
    
    # 诊断结论
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 诊断结论")
    st.markdown(f"**{result.get('diagnosis', 'N/A')}**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 详细发现
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 影像学发现")
        findings = result.get('findings', [])
        for i, finding in enumerate(findings, 1):
            st.markdown(f"{i}. {finding}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 💊 建议处理")
        recommendations = result.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 置信度和严重程度
    col1, col2, col3 = st.columns(3)
    with col1:
        confidence = result.get('confidence', 0)
        st.metric("诊断置信度", f"{confidence:.1%}")
    with col2:
        severity = result.get('severity', 'N/A')
        st.metric("严重程度", severity)
    with col3:
        lesion_count = result.get('lesion_count', 0)
        st.metric("检测病灶数", lesion_count)
    
    # 生成PDF报告
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📄 生成PDF诊断报告", use_container_width=True):
            generate_pdf_report(result)


def generate_pdf_report(result):
    """生成PDF报告"""
    try:
        generator = PDFReportGenerator()
        
        # 创建临时图片（如果有的话）
        temp_images = []
        if st.session_state.get('image_data') is not None:
            import tempfile
            from PIL import Image
            
            image_data = st.session_state['image_data']
            if len(image_data.shape) == 3:
                slice_data = image_data[image_data.shape[0]//2]
            else:
                slice_data = image_data
            
            # 归一化
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
            slice_data = (slice_data * 255).astype(np.uint8)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            Image.fromarray(slice_data).save(temp_file.name)
            temp_images.append(temp_file.name)
        
        # 生成PDF
        pdf_path = generator.generate_report(
            patient_info={
                "patient_id": f"P{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "study_date": datetime.now().strftime("%Y-%m-%d"),
                "modality": "CT"
            },
            diagnosis_result=result,
            image_paths=temp_images
        )
        
        # 提供下载
        with open(pdf_path, 'rb') as f:
            st.download_button(
                "⬇️ 下载PDF报告",
                f.read(),
                file_name=f"RADAR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        st.success("✅ PDF报告生成成功！")
        
    except Exception as e:
        st.error(f"❌ PDF生成失败: {str(e)}")


def batch_processing_page():
    """批量处理页面"""
    st.markdown("## 📊 批量影像处理")
    st.markdown('<div class="info-box">💡 同时处理多个影像文件，提高工作效率</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "上传多个影像文件",
        type=['nii', 'gz', 'dcm', 'npy'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 开始批量处理"):
        processor = BatchProcessor()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, file in enumerate(uploaded_files):
            status_text.text(f"正在处理: {file.name} ({i+1}/{len(uploaded_files)})")
            
            temp_path = save_uploaded_file(file)
            try:
                result = processor.process_single_file(temp_path)
                results.append({
                    'filename': file.name,
                    'status': 'success',
                    'result': result
                })
            except Exception as e:
                results.append({
                    'filename': file.name,
                    'status': 'error',
                    'error': str(e)
                })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        st.session_state['batch_results'] = results
        status_text.text("✅ 批量处理完成！")
        
        # 显示结果
        display_batch_results(results)


def display_batch_results(results):
    """显示批量处理结果"""
    st.markdown("### 📊 批量处理结果")
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总文件数", len(results))
    with col2:
        st.metric("成功", success_count, delta=success_count)
    with col3:
        st.metric("失败", error_count, delta=-error_count if error_count > 0 else 0)
    
    # 详细结果表格
    for result in results:
        with st.expander(f"📄 {result['filename']} - {result['status'].upper()}"):
            if result['status'] == 'success':
                st.json(result['result'])
            else:
                st.error(f"错误: {result['error']}")


def history_page():
    """历史记录页面"""
    st.markdown("## 📜 诊断历史记录")
    
    records = st.session_state['history_manager'].get_all_records()
    
    if not records:
        st.info("暂无历史记录")
        return
    
    st.markdown(f"共有 **{len(records)}** 条诊断记录")
    
    for record in records:
        with st.expander(f"🔍 {record[2]} - {record[5]}"):  # patient_id - timestamp
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**患者ID:** {record[2]}")
                st.markdown(f"**文件路径:** {record[3]}")
                st.markdown(f"**影像尺寸:** {record[6]}")
            with col2:
                st.markdown(f"**诊断时间:** {record[5]}")
                try:
                    result = json.loads(record[4])
                    st.markdown(f"**诊断结论:** {result.get('diagnosis', 'N/A')}")
                except:
                    st.markdown("**诊断结果:** 解析错误")


def settings_page():
    """系统设置页面"""
    st.markdown("## ⚙️ 系统设置")
    
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 🎨 界面设置")
    theme = st.selectbox("主题", ["默认主题", "深色主题", "医疗主题"])
    language = st.selectbox("语言", ["中文", "English"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI模型设置")
    model_path = st.text_input("模型路径", "../pretrained_ckpt")
    use_gpu = st.checkbox("使用GPU加速", value=True)
    confidence_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.7)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 📊 系统信息")
    st.markdown(f"**版本:** v4.0 专业版")
    st.markdown(f"**最后更新:** {datetime.now().strftime('%Y-%m-%d')}")
    st.markdown(f"**Python版本:** {sys.version.split()[0]}")
    st.markdown('</div>', unsafe_allow_html=True)


def save_uploaded_file(uploaded_file):
    """保存上传的文件到临时目录"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


if __name__ == "__main__":
    main()
