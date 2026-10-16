"""
文件上传与格式转换模块
支持NIfTI、DICOM等医学影像格式
"""
import os
import tempfile
import shutil
from pathlib import Path
import SimpleITK as sitk
import nibabel as nib
import numpy as np


class FileHandler:
    """医学影像文件处理器"""
    
    SUPPORTED_FORMATS = {
        'nifti': ['.nii', '.nii.gz'],
        'dicom': ['.dcm', '.dicom'],
        'mhd': ['.mhd', '.mha'],
        'nrrd': ['.nrrd']
    }
    
    def __init__(self, temp_dir=None):
        """初始化文件处理器
        
        Args:
            temp_dir: 临时文件目录，None则使用系统临时目录
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.work_dir = None
        
    def detect_format(self, file_path):
        """检测文件格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 格式类型 ('nifti', 'dicom', 'mhd', 'nrrd', 'unknown')
        """
        if os.path.isdir(file_path):
            # 检查是否是DICOM目录
            dcm_files = [f for f in os.listdir(file_path) 
                        if f.endswith(('.dcm', '.dicom')) or not '.' in f]
            if dcm_files:
                return 'dicom'
            return 'unknown'
        
        file_ext = Path(file_path).suffix.lower()
        if file_path.endswith('.nii.gz'):
            file_ext = '.nii.gz'
            
        for format_type, extensions in self.SUPPORTED_FORMATS.items():
            if file_ext in extensions:
                return format_type
                
        return 'unknown'
    
    def convert_to_nifti(self, input_path, output_path=None):
        """将各种格式转换为NIfTI格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出NIfTI文件路径，None则自动生成
            
        Returns:
            str: 转换后的NIfTI文件路径
        """
        format_type = self.detect_format(input_path)
        
        if output_path is None:
            output_path = os.path.join(
                self.temp_dir, 
                f"converted_{Path(input_path).stem}.nii.gz"
            )
        
        if format_type == 'nifti':
            # 已经是NIfTI格式，直接复制
            if input_path != output_path:
                shutil.copy(input_path, output_path)
            return output_path
            
        elif format_type == 'dicom':
            # DICOM转换
            return self._convert_dicom_to_nifti(input_path, output_path)
            
        elif format_type in ['mhd', 'nrrd']:
            # 使用SimpleITK转换
            image = sitk.ReadImage(input_path)
            sitk.WriteImage(image, output_path)
            return output_path
            
        else:
            raise ValueError(f"不支持的文件格式: {input_path}")
    
    def _convert_dicom_to_nifti(self, dicom_path, output_path):
        """DICOM转NIfTI
        
        Args:
            dicom_path: DICOM文件或目录路径
            output_path: 输出NIfTI文件路径
            
        Returns:
            str: 输出文件路径
        """
        reader = sitk.ImageSeriesReader()
        
        if os.path.isdir(dicom_path):
            # DICOM系列目录
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
            if not dicom_names:
                raise ValueError(f"目录中未找到DICOM文件: {dicom_path}")
            reader.SetFileNames(dicom_names)
        else:
            # 单个DICOM文件
            reader.SetFileNames([dicom_path])
        
        image = reader.Execute()
        sitk.WriteImage(image, output_path)
        return output_path
    
    def load_nifti(self, nifti_path):
        """加载NIfTI文件
        
        Args:
            nifti_path: NIfTI文件路径
            
        Returns:
            tuple: (image_array, affine, header)
        """
        nifti_img = nib.load(nifti_path)
        image_array = nifti_img.get_fdata()
        affine = nifti_img.affine
        header = nifti_img.header
        
        return image_array, affine, header
    
    def get_spacing(self, nifti_path):
        """获取影像的体素间距
        
        Args:
            nifti_path: NIfTI文件路径
            
        Returns:
            tuple: (spacing_x, spacing_y, spacing_z)
        """
        _, affine, _ = self.load_nifti(nifti_path)
        spacing = (
            abs(affine[0, 0]),
            abs(affine[1, 1]),
            abs(affine[2, 2])
        )
        return spacing
    
    def save_nifti(self, image_array, output_path, affine=None, header=None):
        """保存为NIfTI文件
        
        Args:
            image_array: 影像数据数组
            output_path: 输出文件路径
            affine: 仿射矩阵
            header: 头文件信息
        """
        if affine is None:
            affine = np.eye(4)
            
        nifti_img = nib.Nifti1Image(image_array, affine, header)
        nib.save(nifti_img, output_path)
    
    def cleanup(self, path):
        """清理临时文件
        
        Args:
            path: 要删除的文件或目录路径
        """
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)


# 测试代码
if __name__ == "__main__":
    handler = FileHandler()
    print("支持的格式:", handler.SUPPORTED_FORMATS)
