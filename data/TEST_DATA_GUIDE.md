# RADAR测试数据使用指南

## 已生成的测试数据

data文件夹下现在包含以下测试CT数据：

### 1. test_ct_standard.npy (推荐)
- **文件大小**: 16 MB
- **数据形状**: (128, 256, 256)
- **用途**: 标准演示和测试
- **包含结构**:
  - 腹部轮廓（椭圆形身体）
  - 脊柱（高密度区域）
  - 肝脏（右上腹）
  - 双侧肾脏
  - 模拟病灶（用于AI诊断演示）

### 2. test_ct_highres.npy
- **文件大小**: 100 MB
- **数据形状**: (200, 512, 512)
- **用途**: 高分辨率测试，验证系统性能

### 3. test_ct_small.npy
- **文件大小**: 2 MB
- **数据形状**: (64, 128, 128)
- **用途**: 快速加载测试和调试

### 4. test_slice_2d.npy
- **文件大小**: 128 KB
- **数据形状**: (256, 256)
- **用途**: 2D图像查看器测试

## 快速开始

### 方法1: Web界面演示（推荐）

1. **启动Web平台**
   ```bash
   start_web_v4.bat
   ```
   或
   ```bash
   conda activate radar
   cd radar-web
   python -m streamlit run app_v4.py --server.port 8504
   ```

2. **访问系统**
   - 本地地址: http://localhost:8504
   - 网络地址: http://YOUR_IP:8504

3. **上传测试数据**
   - 在"单例诊断"页面
   - 点击"上传医学影像"
   - 选择 `data/test_ct_standard.npy`

4. **查看影像**
   - 选择"单视图高级查看器"（推荐）
   - 使用工具栏调整窗位窗宽
   - 尝试平移、缩放功能
   - 或选择"多视图（三视图）"查看轴位、矢状位、冠状位

5. **AI诊断**
   - 点击"开始AI诊断分析"
   - 等待AI处理
   - 查看诊断结果报告

6. **生成报告**
   - 点击"生成PDF诊断报告"
   - 下载完整诊断报告

### 方法2: Python脚本加载

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载标准测试数据
ct_data = np.load('data/test_ct_standard.npy')

print(f"数据形状: {ct_data.shape}")
print(f"数据类型: {ct_data.dtype}")
print(f"HU值范围: [{ct_data.min()}, {ct_data.max()}]")

# 显示中间切片
mid_slice = ct_data.shape[0] // 2
plt.figure(figsize=(10, 10))
plt.imshow(ct_data[mid_slice], cmap='gray', vmin=-200, vmax=200)
plt.title(f'CT Slice {mid_slice}')
plt.colorbar(label='HU')
plt.axis('off')
plt.show()

# 显示三个方向的切片
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 轴位
axes[0].imshow(ct_data[mid_slice, :, :], cmap='gray')
axes[0].set_title('Axial')
axes[0].axis('off')

# 矢状位
axes[1].imshow(ct_data[:, :, ct_data.shape[2]//2], cmap='gray')
axes[1].set_title('Sagittal')
axes[1].axis('off')

# 冠状位
axes[2].imshow(ct_data[:, ct_data.shape[1]//2, :], cmap='gray')
axes[2].set_title('Coronal')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

## CT数据说明

### HU值（Hounsfield Unit）范围
本测试数据模拟了真实CT的HU值：

- **空气**: -1000 HU
- **脂肪**: -100 ~ -50 HU
- **软组织**: 20 ~ 60 HU
- **肝脏**: 50 ~ 70 HU
- **肾脏**: 30 ~ 40 HU
- **骨骼**: 400 ~ 1000 HU
- **病灶（模拟）**: 100 ~ 150 HU

### 窗位窗宽建议

不同组织查看建议：

| 组织 | 窗位 (WC) | 窗宽 (WW) |
|------|-----------|-----------|
| 腹部软组织 | 40 | 400 |
| 肝脏 | 60 | 150 |
| 骨骼 | 400 | 1500 |
| 肺窗 | -600 | 1500 |

在v4.0专业查看器中，您可以：
- 使用滑块精确调整窗位窗宽
- 使用"窗位/窗宽"工具拖动鼠标调整
- 实时查看调整效果

## 演示功能

使用这些测试数据，您可以演示：

### 1. 专业影像查看
- [x] 三视图显示（轴位、矢状位、冠状位）
- [x] 窗位窗宽调整
- [x] 平移和缩放
- [x] 像素值实时读取
- [x] 切片快速定位

### 2. AI诊断功能
- [x] 自动影像分析
- [x] 病灶检测
- [x] 诊断报告生成
- [x] 置信度评估

### 3. 数据管理
- [x] 历史记录查询
- [x] 批量处理
- [x] PDF报告导出

### 4. 系统性能
- [x] 大文件加载（100MB+）
- [x] 实时渲染
- [x] 响应式交互

## 注意事项

### 这是模拟数据
- **仅用于**: 系统演示、功能测试、用户培训
- **不可用于**: 临床诊断、医学研究、真实患者

### 真实数据
如需使用真实CT数据：
1. `data/demo_cases/AC423ccbe.nii.gz` - 真实病例样本
2. 自备符合格式的医学影像（NIfTI、DICOM等）
3. 注意患者隐私保护和数据安全

### 文件格式支持
系统支持以下格式：
- **.npy**: NumPy数组（本测试数据格式）
- **.nii / .nii.gz**: NIfTI医学影像格式
- **.dcm**: DICOM格式
- **.png / .jpg**: 2D图像格式

## 故障排查

### 上传失败
- 检查文件大小（建议<200MB）
- 确认文件格式正确
- 尝试使用test_ct_small.npy快速测试

### 查看器显示异常
- 刷新浏览器页面
- 清除浏览器缓存
- 检查浏览器控制台错误信息

### AI诊断无响应
- 这是正常的，当前使用模拟推理
- 真实模型需要在conda环境中安装依赖
- 参考 RADAR_inference/inference_demo.py

## 下一步

1. **体验完整功能**: 使用test_ct_standard.npy完整体验所有功能
2. **性能测试**: 使用test_ct_highres.npy测试系统性能
3. **真实数据**: 准备真实CT数据进行测试
4. **集成真实模型**: 参考文档集成RADAR真实推理模型

---

**RADAR医学影像诊断平台 v4.0** | 测试数据使用指南
