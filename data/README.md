# RADAR测试数据集

本文件夹包含用于演示和测试的模拟CT扫描数据。

## 数据文件

### 1. test_ct_standard.npy
- **用途**: 标准测试数据，推荐用于常规演示
- **形状**: (128, 256, 256)
- **说明**: 模拟的腹部CT扫描，包含：
  - 身体轮廓
  - 脊柱（高密度）
  - 肝脏区域
  - 双侧肾脏
  - 模拟病灶

### 2. test_ct_highres.npy
- **用途**: 高分辨率数据测试
- **形状**: (200, 512, 512)
- **说明**: 更高分辨率的CT数据，适合测试系统性能

### 3. test_ct_small.npy
- **用途**: 快速加载测试
- **形状**: (64, 128, 128)
- **说明**: 小尺寸数据，用于快速测试和调试

### 4. test_slice_2d.npy
- **用途**: 2D图像测试
- **形状**: (256, 256)
- **说明**: 单个2D切片，用于测试2D查看器

## 使用方法

### Web界面上传
1. 启动RADAR Web平台：`start_web_v4.bat`
2. 访问 http://localhost:8504
3. 在"单例诊断"页面上传任意.npy文件
4. 使用专业查看器浏览影像
5. 点击"开始AI诊断分析"进行推理

### Python脚本加载
```python
import numpy as np

# 加载数据
ct_data = np.load('data/test_ct_standard.npy')
print(f"数据形状: {ct_data.shape}")
print(f"数据类型: {ct_data.dtype}")
print(f"HU值范围: [{ct_data.min()}, {ct_data.max()}]")

# 查看中间切片
import matplotlib.pyplot as plt
plt.imshow(ct_data[ct_data.shape[0]//2], cmap='gray')
plt.show()
```

## 数据说明

### CT值（HU值）范围
- 空气: -1000 HU
- 脂肪: -100 ~ -50 HU
- 软组织: 20 ~ 60 HU
- 肝脏: 50 ~ 70 HU
- 肾脏: 30 ~ 40 HU
- 骨骼: 400 ~ 1000 HU
- 病灶: 100 ~ 150 HU（模拟）

### 解剖结构
本测试数据模拟包含：
- 腹部轮廓（椭圆形）
- 脊柱（后部中央）
- 肝脏（右上腹）
- 双侧肾脏
- 模拟病灶（用于AI诊断演示）

## 注意事项

⚠️ 这些是**模拟数据**，仅用于：
- 系统功能演示
- 开发测试
- 用户培训

❌ 不能用于：
- 临床诊断
- 医学研究
- 真实患者数据处理

## 真实数据

如需使用真实数据，请参考：
- `data/demo_cases/AC423ccbe.nii.gz` - 真实CT病例（如果可用）
- 自行准备符合格式的医学影像数据
