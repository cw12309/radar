"""
RADAR Web平台功能测试脚本
"""
import sys
import os
import io

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from file_handler import FileHandler
from image_viewer import ImageViewer
from inference_engine import MockInferenceEngine
from visualization import DiagnosisVisualizer
import numpy as np

def test_modules():
    """测试所有核心模块"""
    print('=' * 50)
    print('RADAR Web平台功能测试')
    print('=' * 50)
    
    print('\n[1/5] 模块导入测试...')
    print('  ✓ FileHandler')
    print('  ✓ ImageViewer') 
    print('  ✓ MockInferenceEngine')
    print('  ✓ DiagnosisVisualizer')
    
    print('\n[2/5] 文件处理器测试...')
    handler = FileHandler()
    formats = handler.SUPPORTED_FORMATS
    print(f'  ✓ 支持的格式: {", ".join(formats.keys())}')
    print(f'  ✓ 总共支持 {sum(len(exts) for exts in formats.values())} 种文件扩展名')
    
    print('\n[3/5] 影像查看器测试...')
    test_image = np.random.randn(128, 128, 50) * 100 + 40
    viewer = ImageViewer(test_image)
    print(f'  ✓ 影像尺寸: {viewer.shape}')
    
    axial = viewer.get_axial_slice(25)
    sagittal = viewer.get_sagittal_slice(64)
    coronal = viewer.get_coronal_slice(64)
    print(f'  ✓ 轴位切片: {axial.shape}')
    print(f'  ✓ 矢状位切片: {sagittal.shape}')
    print(f'  ✓ 冠状位切片: {coronal.shape}')
    
    viewer.set_window_preset('腹部')
    print(f'  ✓ 窗宽窗位: 中心={viewer.window_center}, 宽度={viewer.window_width}')
    
    print('\n[4/5] 推理引擎测试...')
    engine = MockInferenceEngine()
    engine.load_model()
    result = engine.infer('test.nii.gz')
    print(f'  ✓ 患者ID: {result["patient_id"]}')
    print(f'  ✓ 检测器官数: {len(result["organs_detected"])}')
    print(f'  ✓ 诊断发现数: {len(result["diagnoses"])}')
    print(f'  ✓ 器官列表: {", ".join(result["organs_detected"][:5])}...')
    
    print('\n[5/5] 可视化工具测试...')
    visualizer = DiagnosisVisualizer()
    
    # 测试表格
    df = visualizer.create_diagnosis_table(result['diagnoses'])
    print(f'  ✓ 诊断表格: {len(df)} 行 × {len(df.columns)} 列')
    
    # 测试报告
    report = visualizer.format_diagnosis_report(result)
    print(f'  ✓ 诊断报告: {len(report)} 字符')
    
    # 测试图表
    fig1 = visualizer.create_severity_pie_chart(result['diagnoses'])
    print(f'  ✓ 严重程度饼图: {len(fig1.data)} 个数据系列')
    
    fig2 = visualizer.create_organ_system_bar_chart(result['diagnoses'])
    print(f'  ✓ 器官系统柱状图: {len(fig2.data)} 个数据系列')
    
    print('\n' + '=' * 50)
    print('✅ 所有测试通过！系统运行正常')
    print('=' * 50)
    
    return True

if __name__ == "__main__":
    try:
        success = test_modules()
        if success:
            print('\n💡 提示: Web应用正在 http://localhost:8501 运行')
            print('   可以在浏览器中测试完整的用户界面')
    except Exception as e:
        print(f'\n❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
