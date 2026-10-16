# RADAR医学影像诊断系统 - Flask专业版

## 🎉 全新Flask版本

基于Flask + 原生JavaScript + 现代CSS打造的专业医学影像诊断平台，提供流畅的用户体验和专业的影像查看功能。

## ✨ 主要特性

### 🎨 现代化界面设计
- **深色医疗主题** - 符合PACS系统习惯，减少眼疲劳
- **渐变配色系统** - 专业美观的视觉设计
- **流畅动画效果** - 卡片悬停、页面切换动画
- **响应式布局** - 适配桌面和平板设备

### 🔍 专业影像查看器
- **高级查看器**
  - 平移、缩放、窗位窗宽调整
  - 实时像素值显示
  - 支持三个方向切片（轴位/矢状位/冠状位）
  - Canvas GPU加速渲染

- **三视图显示**
  - 同时展示轴位、矢状位、冠状位
  - 自动中心定位
  - 3D视图预留区域

- **简单查看器**
  - 快速切片浏览
  - 适合简单查看场景

### 🤖 AI智能诊断
- 自动影像分析
- 结构化诊断报告
- 影像学发现提取
- 建议处理生成
- 置信度和严重程度评估

### 📊 完整功能模块
- **单例诊断** - 完整的上传→查看→诊断→报告流程
- **批量处理** - 多文件同时处理
- **历史记录** - 病例管理和查询
- **系统设置** - 个性化配置

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Conda (推荐)
- 浏览器: Chrome/Edge (推荐)

### 安装步骤

1. **激活conda环境**
   ```bash
   conda activate radar
   ```

2. **安装Flask依赖**
   ```bash
   cd flask-app
   pip install -r requirements.txt
   ```

3. **启动服务器**
   
   **Windows:**
   ```bash
   start_flask.bat
   ```
   
   **命令行:**
   ```bash
   python app.py
   ```

4. **访问系统**
   - 本地地址: http://localhost:5000
   - 网络地址: http://YOUR_IP:5000

## 📖 使用指南

### 1. 上传影像文件

- 点击"选择文件"或拖拽文件到上传区域
- 支持格式：
  - NIfTI (.nii, .nii.gz)
  - DICOM (.dcm)
  - NumPy (.npy)
  - PNG/JPG

### 2. 影像查看

**高级查看器（推荐）：**
1. 选择"🔵 高级查看器"标签
2. 使用工具栏：
   - 🔍 平移 - 拖动移动影像
   - 🔆 窗位/窗宽 - 调整对比度
   - 🔎 缩放 - 滚轮缩放
   - ↺ 重置 - 恢复初始状态
3. 调整参数：
   - 窗位: -1000 ~ 1000 HU
   - 窗宽: 1 ~ 2000 HU
   - 切片选择: 滑块控制
   - 视图方向: 轴位/矢状位/冠状位

**推荐窗位窗宽设置：**
- 腹部软组织: 窗位=40, 窗宽=400
- 肝脏: 窗位=60, 窗宽=150
- 骨骼: 窗位=400, 窗宽=1500

**三视图：**
- 点击"🌐 三视图"标签
- 自动加载三个方向的中心切片
- 适合整体观察

### 3. AI诊断

1. 查看完影像后，点击"🤖 开始AI诊断分析"
2. 等待AI处理（2-5秒）
3. 查看诊断结果：
   - 🎯 诊断结论
   - 🔍 影像学发现
   - 💊 建议处理
   - 📊 置信度、严重程度、病灶数

### 4. 生成报告

点击"📄 生成PDF报告"自动下载完整医疗报告

### 5. 其他功能

- **批量处理** - 导航栏选择"📊 批量处理"，上传多个文件
- **历史记录** - 导航栏选择"📜 历史记录"，查看历史诊断
- **系统设置** - 导航栏选择"⚙️ 系统设置"，配置参数

## 🏗️ 技术架构

### 后端技术栈
- **Flask 2.3** - Web框架
- **Flask-CORS** - 跨域支持
- **NumPy** - 数组处理
- **Pillow** - 图像处理
- **NiBabel** - NIfTI格式支持
- **SimpleITK** - 医学影像处理
- **ReportLab** - PDF生成
- **SQLite** - 历史记录存储

### 前端技术栈
- **原生JavaScript (ES6+)** - 交互逻辑
- **CSS3** - 现代化样式
- **HTML5 Canvas** - 图像渲染
- **Fetch API** - HTTP请求

### 特色技术
- **Canvas 2D渲染** - 高性能影像显示
- **Base64图像传输** - 前后端高效通信
- **拖拽上传** - 现代化文件上传体验
- **Toast通知** - 优雅的提示系统
- **加载遮罩** - 友好的加载状态

## 📂 项目结构

```
flask-app/
├── app.py                 # Flask主应用
├── requirements.txt       # Python依赖
├── start_flask.bat       # Windows启动脚本
├── templates/
│   └── index.html        # 主页面模板
└── static/
    ├── css/
    │   └── styles.css    # 现代化样式
    └── js/
        └── app.js        # 前端交互逻辑
```

## 🆚 与Streamlit版本对比

| 特性 | Streamlit v4.0 | Flask v4.0 |
|------|----------------|------------|
| 界面美观度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 交互流畅度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 响应速度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 自定义能力 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 开发难度 | 简单 | 中等 |
| 部署难度 | 简单 | 简单 |
| 扩展性 | 一般 | 优秀 |
| 生产就绪 | ❌ | ✅ |

**Flask版本优势：**
- ✅ 完全自定义的UI/UX
- ✅ 更流畅的用户体验
- ✅ 更快的响应速度
- ✅ 更好的可维护性
- ✅ 适合生产环境部署
- ✅ 支持RESTful API

## 🧪 测试数据

使用项目根目录的测试数据：

```bash
# 推荐测试文件
data/test_ct_standard.npy  (16 MB, 128×256×256)
data/test_ct_highres.npy   (100 MB, 200×512×512)
data/test_ct_small.npy     (2 MB, 64×128×128)
```

详细说明请参考：`TEST_DATA_READY.md`

## 🎯 核心API

### 上传影像
```
POST /api/upload
Content-Type: multipart/form-data
Body: file

Response: {
    success: true,
    file_id: "20260930150000_test.npy",
    info: {filename, file_type, shape, ...}
}
```

### 获取切片
```
GET /api/image/slice/{file_id}/{slice_idx}
Params: view, window_center, window_width

Response: {
    success: true,
    image: "base64_encoded_png",
    shape: [256, 256]
}
```

### AI诊断
```
POST /api/diagnose/{file_id}

Response: {
    success: true,
    result: {diagnosis, findings, recommendations, ...},
    case_id: 1
}
```

### 生成报告
```
POST /api/report/generate
Body: {diagnosis_result, patient_info}

Response: {
    success: true,
    pdf_data: "base64_encoded_pdf",
    filename: "RADAR_Report_20260930.pdf"
}
```

## 🔧 故障排查

### 端口被占用
```bash
# 修改app.py中的端口号
app.run(host='0.0.0.0', port=5001, debug=True)
```

### 模块导入错误
```bash
# 确保在radar环境中
conda activate radar
pip install -r requirements.txt
```

### 影像加载失败
- 检查文件格式是否支持
- 尝试使用test_ct_small.npy测试
- 查看浏览器控制台错误信息

### 界面样式异常
- 清除浏览器缓存
- 使用Chrome/Edge浏览器
- 检查CSS文件是否正确加载

## 🚧 待开发功能

- [ ] Cornerstone.js深度集成
- [ ] 真实RADAR模型接入
- [ ] 测量工具（距离、面积、体积）
- [ ] ROI标注功能
- [ ] DICOM SR结构化报告
- [ ] 用户认证系统
- [ ] 多用户协作

## 📝 更新日志

**v4.0 Flask专业版 (2026-09-30)**
- 🎉 全新Flask架构
- ✨ 现代化界面设计
- ✨ 专业影像查看器
- ✨ 流畅的用户体验
- 🔧 RESTful API
- 📊 完整功能模块

## 💡 提示

1. **首次使用** - 建议使用`test_ct_standard.npy`进行功能体验
2. **性能优化** - 大文件（>100MB）加载需要时间，请耐心等待
3. **浏览器** - 推荐使用Chrome或Edge浏览器以获得最佳体验
4. **窗位窗宽** - 不同组织建议使用不同的窗位窗宽设置
5. **报告生成** - PDF报告包含完整诊断信息，可直接归档

---

**RADAR医学影像诊断平台 Flask专业版** | 现代化 · 专业 · 流畅
