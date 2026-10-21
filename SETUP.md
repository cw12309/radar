# 部署与启动指南（SETUP）

面向新机器从零拉起「RADAR 医学影像诊断平台」（Flask 后端 + 交互看片 + OHIF Viewer）。
全程约三步：装依赖 → 下模型权重 → 启动。

## 0. 环境要求

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | 3.10（conda 环境名建议 `radar`） | 启动脚本通过 `conda run -n radar` 调用 |
| Node.js | >= 18 | OHIF Viewer 构建需要 |
| yarn | 1.22.x | OHIF 依赖安装（`npm i -g yarn`） |
| Git | 任意新版 | 克隆仓库 |
| Git LFS | 任意新版 | 仓库二进制资源走 LFS，克隆前需 `git lfs install` |
| 显卡 | 可选 | 有 NVIDIA GPU 自动用 CUDA，否则 CPU 推理（较慢） |

## 1. 克隆仓库

仓库的二进制资源（演示 DICOM、npy 测试数据、png 等）通过 **Git LFS** 存储，
克隆前请先安装并初始化 LFS，否则只会拿到指针文件：

```bash
git lfs install          # 一次性；未安装时: https://git-lfs.com 或 winget install GitHub.GitLFS
git clone https://github.com/cw12309/radar.git
cd radar
```

仓库**不含**模型权重与大数据集（见第 3、5 节），克隆体积约 200MB。

## 2. Python 依赖

```bash
conda create -n radar python=3.10 -y
conda activate radar

# 官方 RADAR 依赖 + 本平台后端/推理依赖
pip install -r requirements.txt
pip install -r flask-app/requirements.txt
```

GPU 机器建议把 torch 换成 CUDA 版（以 cu121 为例，按驱动选择）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> 已验证组合：torch 2.x + transformers 4.25 + timm 0.4.12 + monai 1.6。
> `transformers`/`timm`/`fairscale` 版本必须与 `requirements.txt` 一致，否则 RADAR 模型加载会报错。

## 3. 下载模型权重（约 3.5GB，来自 HuggingFace 官方仓库）

```bash
python download_scripts/download_checkpoints.py --local-dir ckpt
```

会下载 `checkpoint_radar_pretrain.pth`、`checkpoint_unet.pth`、`bert-base-chinese/`、`bert-base-uncased/` 等。
推理必需但 **HF 仓库没有** 的 `infer_text_embedding_radar.pt` 已随本仓库分发，无需额外操作。

下载完成后 `ckpt/` 应至少包含：

```
ckpt/
├── checkpoint_radar_pretrain.pth      # 推理主权重（必需）
├── infer_text_embedding_radar.pt      # 文本嵌入（仓库自带，必需）
├── bert-base-chinese/                 # 中文 BERT（必需）
└── bert-base-uncased/                 # 英文 BERT（必需）
```

## 4. OHIF Viewer 依赖（首次约 10 分钟）

```bash
cd ohif-viewer
yarn install
cd ..
```

## 5. 测试数据

- 仓库自带内置演示 DICOM 序列 `data/test_ct_dicom_series/`（OHIF 演示用，无需下载）。
- 单文件体验可用 `data/test_ct_standard.npy`（16MB，仓库自带）。
- 真实病例：直接在网页上传 NIfTI / DICOM 文件夹即可。
  > Windows 注意：Chrome 受 260 字符路径限制，DICOM 文件夹路径过长会上传失败，
  > 请把数据放到短路径（如 `D:\dcm\`）再选择上传。

## 5.5 可选：精细分割（TotalSegmentator）

2D 叠加与 3D 重建默认使用 RADAR 模型自带的分割掩码（前端「分割来源」可切
「模型原始输出」查看未补全的原始边界）。若想要更精细的器官边界（原始
DICOM 分辨率），可安装独立的 TotalSegmentator 环境。**注意：CPU 推理单次
需数十分钟，默认关闭**；设置 `RADAR_ENABLE_FINE_SEG=1` 后，诊断完成才会
后台自动跑精细分割，完成后前端「分割来源」出现「精细分割」选项。不装也
完全不影响诊断与默认显示。

```powershell
conda create -n radar-ts python=3.10 -y
$env:PYTHONNOUSERSITE='1'                       # 隔离用户级 site-packages
& "$env:USERPROFILE\.conda\envs\radar-ts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& "$env:USERPROFILE\.conda\envs\radar-ts\python.exe" -m pip install TotalSegmentator
```

> 若 `TotalSegmentator` 安装时 dipy 编译失败，先
> `pip download dipy==1.9.0 --no-deps -d wheels` 再 `pip install wheels\dipy-*.whl`。
> 首次运行会自动下载 TS 权重（约 200MB）。

## 6. 启动

Windows（推荐）：

```powershell
.\start_radar_system.bat          # 或 .\start_radar_system.ps1
.\start_radar_system.ps1 -Stop    # 停止
```

脚本会拉起 Flask（5000）与 OHIF（3001），日志在 `logs/`。

Linux / macOS 手动启动：

```bash
conda activate radar
python flask-app/app.py &                 # Flask :5000
cd ohif-viewer && OHIF_PORT=3001 npm run dev &   # OHIF :3001
```

打开 <http://127.0.0.1:5000> 即可使用：上传 → AI 诊断 → 交互看片（分割叠加 / 注意力热力）→ 导出 PDF 报告。

## 7. 常见问题

| 现象 | 处理 |
| --- | --- |
| 启动报 `conda was not found` | 把 conda 加入 PATH，或改手动启动（见第 6 节） |
| 诊断报 `未找到RADAR权重文件` | 第 3 步没做完，检查 `ckpt/checkpoint_radar_pretrain.pth` |
| 诊断报 `缺少文本嵌入文件` | `ckpt/infer_text_embedding_radar.pt` 缺失（git 拉取不完整） |
| OHIF 页面打不开 | 等 `yarn install` 与首次 webpack 编译完成（日志 `logs/ohif.log`） |
| PDF 中文乱码（Linux） | 系统缺中文字体；Windows 自带微软雅黑无需处理 |
| 上传 DICOM 报 `Failed to fetch` | 见第 5 节路径长度提示 |
| 克隆后 npy/dcm 文件只有几百字节 | 没装 Git LFS，执行 `git lfs install` 后 `git lfs pull` |

## 8. 目录速览

```
flask-app/        Flask 后端 + 前端页面（交互看片、PDF 报告）
radar-web/        推理引擎、PDF 生成器、历史管理等模块
RADAR_inference/  官方 RADAR 推理代码（被推理引擎复用）
RADAR_train/      官方训练代码
ohif-viewer/      OHIF Viewer 源码（yarn 安装依赖）
download_scripts/ 权重下载脚本
data/             内置演示序列与小测试数据
ckpt/             模型权重（下载，不入库）
docs/             官方推理/训练/预处理文档
```
