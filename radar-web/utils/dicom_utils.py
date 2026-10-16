"""
DICOM工具模块
处理DICOM系列文件的读取、元数据提取等
"""
import os
import pydicom
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple


class DICOMHandler:
    """DICOM文件处理器"""
    
    def __init__(self):
        pass
    
    def is_dicom_file(self, file_path: str) -> bool:
        """判断是否为DICOM文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否为DICOM文件
        """
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
        except:
            return False
    
    def read_dicom_series(self, directory: str) -> Tuple[np.ndarray, Dict]:
        """读取DICOM系列
        
        Args:
            directory: DICOM文件目录
            
        Returns:
            tuple: (image_array, metadata_dict)
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        
        if not dicom_names:
            raise ValueError(f"目录中未找到DICOM系列: {directory}")
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # 转换为numpy数组
        image_array = sitk.GetArrayFromImage(image)
        
        # 读取元数据
        metadata = self.extract_metadata(dicom_names[0])
        metadata['series_files_count'] = len(dicom_names)
        
        # 获取spacing
        spacing = image.GetSpacing()
        metadata['spacing'] = spacing
        
        return image_array, metadata
    
    def extract_metadata(self, dicom_file: str) -> Dict:
        """提取DICOM元数据
        
        Args:
            dicom_file: DICOM文件路径
            
        Returns:
            dict: 元数据字典
        """
        ds = pydicom.dcmread(dicom_file)
        
        metadata = {
            'patient_name': str(getattr(ds, 'PatientName', 'Unknown')),
            'patient_id': str(getattr(ds, 'PatientID', 'Unknown')),
            'patient_age': str(getattr(ds, 'PatientAge', 'Unknown')),
            'patient_sex': str(getattr(ds, 'PatientSex', 'Unknown')),
            'study_date': str(getattr(ds, 'StudyDate', 'Unknown')),
            'study_time': str(getattr(ds, 'StudyTime', 'Unknown')),
            'study_description': str(getattr(ds, 'StudyDescription', 'Unknown')),
            'series_description': str(getattr(ds, 'SeriesDescription', 'Unknown')),
            'modality': str(getattr(ds, 'Modality', 'Unknown')),
            'manufacturer': str(getattr(ds, 'Manufacturer', 'Unknown')),
            'institution_name': str(getattr(ds, 'InstitutionName', 'Unknown')),
            'slice_thickness': str(getattr(ds, 'SliceThickness', 'Unknown')),
            'kvp': str(getattr(ds, 'KVP', 'Unknown')),
            'exposure': str(getattr(ds, 'Exposure', 'Unknown')),
        }
        
        # 图像尺寸信息
        if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
            metadata['image_size'] = f"{ds.Rows} x {ds.Columns}"
        
        return metadata
    
    def anonymize_metadata(self, metadata: Dict) -> Dict:
        """脱敏元数据（移除患者隐私信息）
        
        Args:
            metadata: 原始元数据
            
        Returns:
            dict: 脱敏后的元数据
        """
        anonymized = metadata.copy()
        
        # 移除隐私字段
        privacy_fields = ['patient_name', 'patient_id']
        for field in privacy_fields:
            if field in anonymized:
                anonymized[field] = '***'
        
        return anonymized
    
    def format_metadata_display(self, metadata: Dict, anonymize: bool = True) -> str:
        """格式化元数据为可读文本
        
        Args:
            metadata: 元数据字典
            anonymize: 是否脱敏
            
        Returns:
            str: 格式化后的文本
        """
        if anonymize:
            metadata = self.anonymize_metadata(metadata)
        
        display_text = "DICOM影像信息\n"
        display_text += "=" * 50 + "\n\n"
        
        display_text += "患者信息:\n"
        display_text += f"  姓名: {metadata.get('patient_name', 'Unknown')}\n"
        display_text += f"  ID: {metadata.get('patient_id', 'Unknown')}\n"
        display_text += f"  年龄: {metadata.get('patient_age', 'Unknown')}\n"
        display_text += f"  性别: {metadata.get('patient_sex', 'Unknown')}\n\n"
        
        display_text += "检查信息:\n"
        display_text += f"  日期: {metadata.get('study_date', 'Unknown')}\n"
        display_text += f"  时间: {metadata.get('study_time', 'Unknown')}\n"
        display_text += f"  检查描述: {metadata.get('study_description', 'Unknown')}\n"
        display_text += f"  序列描述: {metadata.get('series_description', 'Unknown')}\n"
        display_text += f"  模态: {metadata.get('modality', 'Unknown')}\n\n"
        
        display_text += "设备信息:\n"
        display_text += f"  厂商: {metadata.get('manufacturer', 'Unknown')}\n"
        display_text += f"  机构: {metadata.get('institution_name', 'Unknown')}\n\n"
        
        display_text += "采集参数:\n"
        display_text += f"  层厚: {metadata.get('slice_thickness', 'Unknown')} mm\n"
        display_text += f"  管电压: {metadata.get('kvp', 'Unknown')} kV\n"
        display_text += f"  曝光量: {metadata.get('exposure', 'Unknown')} mAs\n"
        
        if 'series_files_count' in metadata:
            display_text += f"  文件数: {metadata['series_files_count']}\n"
        
        if 'spacing' in metadata:
            spacing = metadata['spacing']
            display_text += f"  体素间距: {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm\n"
        
        return display_text
    
    def find_dicom_series(self, directory: str) -> List[str]:
        """在目录中查找所有DICOM系列
        
        Args:
            directory: 搜索目录
            
        Returns:
            list: 系列UID列表
        """
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(directory)
        return series_ids


# 测试代码
if __name__ == "__main__":
    handler = DICOMHandler()
    print("DICOM处理器初始化成功")
    print(f"支持的功能:")
    print("  - DICOM系列读取")
    print("  - 元数据提取")
    print("  - 隐私脱敏")
