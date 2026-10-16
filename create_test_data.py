"""
创建测试CT扫描数据
生成一个完整的腹部CT模拟数据用于演示
"""
import numpy as np
import os
from pathlib import Path

def create_synthetic_ct_scan(output_path, shape=(128, 256, 256)):
    """
    创建模拟的腹部CT扫描数据
    
    Args:
        output_path: 输出文件路径
        shape: 数据形状 (depth, height, width)
    """
    print(f"正在生成模拟CT数据，形状: {shape}")
    
    depth, height, width = shape
    
    # 创建基础CT数据 (HU值范围: -1000 到 3000)
    ct_data = np.zeros(shape, dtype=np.int16)
    
    # 背景空气 (-1000 HU)
    ct_data[:] = -1000
    
    # 创建身体轮廓（椭圆形）
    z_center = depth // 2
    y_center = height // 2
    x_center = width // 2
    
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # 身体主体（软组织，约40 HU）
                dy = (y - y_center) / (height * 0.35)
                dx = (x - x_center) / (width * 0.35)
                
                if dy**2 + dx**2 < 1:
                    ct_data[z, y, x] = 40 + np.random.randint(-20, 20)
                    
                    # 添加脊柱（骨骼，约400-800 HU）
                    if abs(x - x_center) < width * 0.08 and y > y_center + height * 0.1:
                        ct_data[z, y, x] = 600 + np.random.randint(-100, 100)
                    
                    # 添加肝脏（约50-70 HU）
                    if (x > x_center + width * 0.05 and x < x_center + width * 0.25 and
                        y > y_center - height * 0.15 and y < y_center + height * 0.1 and
                        z > depth * 0.3 and z < depth * 0.6):
                        ct_data[z, y, x] = 60 + np.random.randint(-10, 10)
                    
                    # 添加肾脏（约30-40 HU）
                    # 左肾
                    if (abs(x - (x_center - width * 0.15)) < width * 0.05 and
                        abs(y - (y_center + height * 0.05)) < height * 0.08 and
                        z > depth * 0.4 and z < depth * 0.6):
                        ct_data[z, y, x] = 35 + np.random.randint(-5, 5)
                    
                    # 右肾
                    if (abs(x - (x_center + width * 0.15)) < width * 0.05 and
                        abs(y - (y_center + height * 0.05)) < height * 0.08 and
                        z > depth * 0.4 and z < depth * 0.6):
                        ct_data[z, y, x] = 35 + np.random.randint(-5, 5)
                    
                    # 添加模拟病灶（高密度，约100-150 HU）
                    lesion_x = x_center + int(width * 0.15)
                    lesion_y = y_center - int(height * 0.05)
                    lesion_z_start = int(depth * 0.45)
                    lesion_z_end = int(depth * 0.55)
                    
                    if (z >= lesion_z_start and z <= lesion_z_end):
                        dist = np.sqrt((x - lesion_x)**2 + (y - lesion_y)**2)
                        if dist < width * 0.04:
                            ct_data[z, y, x] = 120 + np.random.randint(-10, 10)
    
    # 保存为npy格式
    np.save(output_path, ct_data)
    print(f"[OK] 数据已保存到: {output_path}")
    print(f"   形状: {ct_data.shape}")
    print(f"   数据类型: {ct_data.dtype}")
    print(f"   HU值范围: [{ct_data.min()}, {ct_data.max()}]")
    print(f"   文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return ct_data


def create_test_dataset():
    """创建完整的测试数据集"""
    
    # 创建data文件夹
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("创建RADAR测试数据集")
    print("=" * 60)
    print()
    
    # 1. 标准CT扫描
    print("1. 生成标准腹部CT扫描...")
    ct_standard = create_synthetic_ct_scan(
        data_dir / "test_ct_standard.npy",
        shape=(128, 256, 256)
    )
    print()
    
    # 2. 高分辨率CT
    print("2. 生成高分辨率CT扫描...")
    ct_hires = create_synthetic_ct_scan(
        data_dir / "test_ct_highres.npy",
        shape=(200, 512, 512)
    )
    print()
    
    # 3. 小尺寸快速测试
    print("3. 生成快速测试数据...")
    ct_small = create_synthetic_ct_scan(
        data_dir / "test_ct_small.npy",
        shape=(64, 128, 128)
    )
    print()
    
    # 4. 创建2D切片样本
    print("4. 生成2D切片样本...")
    slice_2d = ct_standard[ct_standard.shape[0] // 2, :, :]
    np.save(data_dir / "test_slice_2d.npy", slice_2d)
    print(f"[OK] 2D切片已保存，形状: {slice_2d.shape}")
    print()
    
    # 创建README
    readme_content = """# RADAR测试数据集

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
"""
    
    with open(data_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("=" * 60)
    print("[OK] 测试数据集创建完成！")
    print("=" * 60)
    print()
    print("📁 数据文件位置: data/")
    print()
    print("📋 可用文件:")
    print("   1. test_ct_standard.npy  - 推荐用于演示")
    print("   2. test_ct_highres.npy   - 高分辨率测试")
    print("   3. test_ct_small.npy     - 快速测试")
    print("   4. test_slice_2d.npy     - 2D切片")
    print()
    print("🚀 使用方法:")
    print("   1. 运行 start_web_v4.bat 启动Web平台")
    print("   2. 访问 http://localhost:8504")
    print("   3. 上传 data/test_ct_standard.npy")
    print("   4. 选择'单视图高级查看器'浏览影像")
    print("   5. 点击'开始AI诊断分析'")
    print()


if __name__ == "__main__":
    create_test_dataset()
