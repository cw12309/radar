"""
快速验证测试数据
验证生成的CT数据可以正常加载和显示
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def verify_test_data():
    """验证所有测试数据文件"""
    
    print("=" * 60)
    print("RADAR测试数据验证")
    print("=" * 60)
    print()
    
    data_dir = Path("data")
    test_files = [
        "test_ct_standard.npy",
        "test_ct_highres.npy",
        "test_ct_small.npy",
        "test_slice_2d.npy"
    ]
    
    results = []
    
    for filename in test_files:
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"[FAIL] {filename} - 文件不存在")
            results.append(False)
            continue
        
        try:
            # 加载数据
            data = np.load(filepath)
            
            # 验证信息
            file_size_mb = filepath.stat().st_size / 1024 / 1024
            
            print(f"[OK] {filename}")
            print(f"     形状: {data.shape}")
            print(f"     数据类型: {data.dtype}")
            print(f"     值范围: [{data.min()}, {data.max()}]")
            print(f"     文件大小: {file_size_mb:.2f} MB")
            print()
            
            results.append(True)
            
        except Exception as e:
            print(f"[FAIL] {filename} - 错误: {str(e)}")
            results.append(False)
    
    # 总结
    print("=" * 60)
    success_count = sum(results)
    total_count = len(results)
    
    if success_count == total_count:
        print(f"[OK] 验证完成: {success_count}/{total_count} 个文件正常")
        print()
        print("现在可以:")
        print("  1. 运行 start_web_v4.bat 启动Web平台")
        print("  2. 访问 http://localhost:8504")
        print("  3. 上传 data/test_ct_standard.npy")
        print("  4. 开始体验专业影像查看和AI诊断功能")
    else:
        print(f"[WARN] 验证完成: {success_count}/{total_count} 个文件正常")
        print("请检查失败的文件并重新生成")
    
    print("=" * 60)
    
    return all(results)


def visualize_standard_data():
    """可视化标准测试数据"""
    
    print()
    print("正在生成数据可视化预览...")
    
    try:
        # 加载标准数据
        data = np.load("data/test_ct_standard.npy")
        
        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('RADAR测试CT数据预览 - test_ct_standard.npy', fontsize=16, fontweight='bold')
        
        # 获取中心切片索引
        mid_z = data.shape[0] // 2
        mid_y = data.shape[1] // 2
        mid_x = data.shape[2] // 2
        
        # 轴位（Axial）
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(data[mid_z, :, :], cmap='gray', vmin=-200, vmax=200)
        ax1.set_title(f'Axial View (Slice {mid_z})')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='HU')
        
        # 矢状位（Sagittal）
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(data[:, :, mid_x], cmap='gray', vmin=-200, vmax=200)
        ax2.set_title(f'Sagittal View (Slice {mid_x})')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, label='HU')
        
        # 冠状位（Coronal）
        ax3 = plt.subplot(2, 3, 3)
        im3 = ax3.imshow(data[:, mid_y, :], cmap='gray', vmin=-200, vmax=200)
        ax3.set_title(f'Coronal View (Slice {mid_y})')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, label='HU')
        
        # 病灶区域放大（轴位）
        lesion_z = int(data.shape[0] * 0.5)
        lesion_y_start = int(data.shape[1] * 0.4)
        lesion_y_end = int(data.shape[1] * 0.6)
        lesion_x_start = int(data.shape[2] * 0.5)
        lesion_x_end = int(data.shape[2] * 0.7)
        
        ax4 = plt.subplot(2, 3, 4)
        lesion_roi = data[lesion_z, lesion_y_start:lesion_y_end, lesion_x_start:lesion_x_end]
        im4 = ax4.imshow(lesion_roi, cmap='gray', vmin=-200, vmax=200)
        ax4.set_title('病灶区域放大')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, label='HU')
        
        # HU值直方图
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(data.flatten(), bins=100, range=(-1000, 500), color='steelblue', alpha=0.7)
        ax5.set_xlabel('HU Value')
        ax5.set_ylabel('Frequency')
        ax5.set_title('HU值分布直方图')
        ax5.grid(True, alpha=0.3)
        
        # 添加组织标记
        ax5.axvline(-1000, color='cyan', linestyle='--', alpha=0.5, label='空气')
        ax5.axvline(40, color='red', linestyle='--', alpha=0.5, label='软组织')
        ax5.axvline(120, color='orange', linestyle='--', alpha=0.5, label='病灶')
        ax5.axvline(600, color='white', linestyle='--', alpha=0.5, label='骨骼')
        ax5.legend()
        
        # 数据信息
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        info_text = f"""
数据信息:
━━━━━━━━━━━━━━━━━━━━
文件: test_ct_standard.npy
形状: {data.shape}
数据类型: {data.dtype}
HU范围: [{data.min()}, {data.max()}]

包含结构:
• 身体轮廓（椭圆形）
• 脊柱（高密度）
• 肝脏（右上腹）
• 双侧肾脏
• 模拟病灶

建议窗位窗宽:
• 腹部: WC=40, WW=400
• 肝脏: WC=60, WW=150
• 骨骼: WC=400, WW=1500
        """
        
        ax6.text(0.1, 0.5, info_text, 
                fontsize=10, 
                family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        # 保存图像
        output_path = "data/test_data_preview.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] 预览图已保存到: {output_path}")
        
        # 显示图像
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 可视化失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 验证数据
    success = verify_test_data()
    
    if success:
        print()
        response = input("是否生成数据可视化预览? (y/n): ")
        if response.lower() == 'y':
            visualize_standard_data()
