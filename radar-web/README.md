# RADAR医学影像诊断Web平台

基于阿里巴巴达摩院RADAR模型的腹部CT智能诊断Web界面。

## 功能特性

### Phase 1 (当前版本)
- ✅ 多格式影像上传（NIfTI, DICOM, MHD, NRRD）
- ✅ 2D切片查看器（轴位、矢状位、冠状位三视图）
- ✅ 窗宽窗位调整（腹部窗、肺窗、骨窗等预设）
- ✅ 模型推理（支持模拟模式和实际推理）
- ✅ 诊断结果展示（表格、图表、报告）
- ✅ 报告下载（JSON、Markdown格式）

### 规划中
- Phase 2: DICOM完整支持、3D可视化、器官分割叠加
- Phase 3: PDF报告生成、批量处理、历史记录管理

## 安装依赖

```bash
# 安装基础依赖
pip install -r ../requirements.txt

# 安装Web界面额外依赖
pip install -r requirements_web.txt
```

## 运行应用

```bash
# 开发模式（使用模拟推理）
streamlit run app.py

# 生产模式（需要先下载模型）
streamlit run app.py -- --use-real-model
```

访问地址: http://localhost:8501

## 项目结构

```
radar-web/
├── app.py                    # Streamlit主应用
├── requirements_web.txt      # Web界面依赖
├── modules/                  # 核心模块
│   ├── __init__.py
│   ├── file_handler.py       # 文件处理
│   ├── image_viewer.py       # 影像查看器
│   ├── inference_engine.py   # 推理引擎
│   └── visualization.py      # 可视化工具
├── utils/                    # 工具函数
│   └── __init__.py
└── assets/                   # 静态资源
```

## 使用说明

1. **上传影像**: 点击左侧"上传影像"，选择CT影像文件
2. **查看影像**: 在"影像查看"标签页，使用滑块浏览不同切片，调整窗宽窗位
3. **开始诊断**: 点击"开始诊断"按钮，等待推理完成
4. **查看报告**: 在"诊断报告"标签页查看详细诊断结果
5. **统计分析**: 在"统计分析"标签页查看可视化图表
6. **下载报告**: 支持下载JSON和Markdown格式的诊断报告

## 技术栈

- **前端框架**: Streamlit
- **影像处理**: nibabel, SimpleITK, pydicom
- **可视化**: matplotlib, plotly
- **深度学习**: PyTorch, MONAI
- **NLP**: transformers (BERT)

## 注意事项

- 首次使用建议开启"模拟推理"模式进行界面测试
- 实际推理需要确保模型文件已下载到 `../ckpt` 目录
- 上传的影像文件会临时存储，推理后自动清理
- 本系统仅供研究和参考，不能替代专业医师诊断

## 开发者

阿里巴巴达摩院 & Codex AI助手

## 许可证

参见主项目LICENSE文件
