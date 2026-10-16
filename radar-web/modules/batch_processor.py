"""
批量处理模块
支持多文件上传和批量推理处理
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Callable, Optional
from datetime import datetime
import concurrent.futures
from tqdm import tqdm


class BatchProcessor:
    """批量影像处理器"""
    
    def __init__(self, inference_engine, max_workers: int = 3):
        """初始化批量处理器
        
        Args:
            inference_engine: 推理引擎实例
            max_workers: 最大并行worker数
        """
        self.inference_engine = inference_engine
        self.max_workers = max_workers
        self.results = []
        
    def process_directory(self,
                         directory: str,
                         file_pattern: str = "*.nii.gz",
                         progress_callback: Optional[Callable] = None) -> List[Dict]:
        """批量处理目录中的文件
        
        Args:
            directory: 输入目录
            file_pattern: 文件匹配模式
            progress_callback: 进度回调函数
            
        Returns:
            list: 处理结果列表
        """
        # 查找所有匹配文件
        files = list(Path(directory).glob(file_pattern))
        
        if not files:
            raise ValueError(f"在目录 {directory} 中未找到匹配 {file_pattern} 的文件")
        
        print(f"找到 {len(files)} 个文件待处理")
        
        results = []
        
        # 批量处理
        for i, file_path in enumerate(files):
            try:
                if progress_callback:
                    progress_callback(i, len(files), str(file_path))
                
                # 执行推理
                result = self.inference_engine.infer(str(file_path))
                result['file_path'] = str(file_path)
                result['processed_time'] = datetime.now().isoformat()
                results.append(result)
                
                print(f"✓ 完成 {i+1}/{len(files)}: {file_path.name}")
                
            except Exception as e:
                print(f"✗ 处理失败 {file_path.name}: {e}")
                results.append({
                    'file_path': str(file_path),
                    'error': str(e),
                    'status': 'failed'
                })
        
        self.results = results
        return results
    
    def process_file_list(self,
                         file_paths: List[str],
                         progress_callback: Optional[Callable] = None) -> List[Dict]:
        """批量处理文件列表
        
        Args:
            file_paths: 文件路径列表
            progress_callback: 进度回调函数
            
        Returns:
            list: 处理结果列表
        """
        results = []
        
        for i, file_path in enumerate(file_paths):
            try:
                if progress_callback:
                    progress_callback(i, len(file_paths), file_path)
                
                # 执行推理
                result = self.inference_engine.infer(file_path)
                result['file_path'] = file_path
                result['processed_time'] = datetime.now().isoformat()
                results.append(result)
                
            except Exception as e:
                results.append({
                    'file_path': file_path,
                    'error': str(e),
                    'status': 'failed'
                })
        
        self.results = results
        return results
    
    def process_parallel(self,
                        file_paths: List[str],
                        progress_callback: Optional[Callable] = None) -> List[Dict]:
        """并行批量处理（适用于多核CPU）
        
        Args:
            file_paths: 文件路径列表
            progress_callback: 进度回调函数
            
        Returns:
            list: 处理结果列表
        """
        results = []
        
        def process_single(file_path):
            try:
                result = self.inference_engine.infer(file_path)
                result['file_path'] = file_path
                result['processed_time'] = datetime.now().isoformat()
                return result
            except Exception as e:
                return {
                    'file_path': file_path,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single, fp): fp for fp in file_paths}
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(file_paths), result['file_path'])
        
        self.results = results
        return results
    
    def export_results(self, output_path: str, format: str = 'json'):
        """导出批量处理结果
        
        Args:
            output_path: 输出文件路径
            format: 输出格式 ('json', 'csv')
        """
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        elif format == 'csv':
            import pandas as pd
            
            # 展平结果为表格格式
            flat_results = []
            for result in self.results:
                flat_result = {
                    'file_path': result.get('file_path', ''),
                    'patient_id': result.get('patient_id', ''),
                    'organs_count': len(result.get('organs_detected', [])),
                    'diagnoses_count': len(result.get('diagnoses', [])),
                    'status': result.get('status', 'success'),
                    'processed_time': result.get('processed_time', '')
                }
                
                # 添加诊断信息
                diagnoses = result.get('diagnoses', [])
                if diagnoses:
                    flat_result['findings'] = '; '.join([
                        f"{d['organ']}: {d['finding']}" for d in diagnoses
                    ])
                else:
                    flat_result['findings'] = '无异常'
                
                flat_results.append(flat_result)
            
            df = pd.DataFrame(flat_results)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"结果已导出到: {output_path}")
    
    def generate_summary_report(self) -> Dict:
        """生成批量处理摘要报告
        
        Returns:
            dict: 摘要报告
        """
        if not self.results:
            return {}
        
        total = len(self.results)
        success = len([r for r in self.results if r.get('status') != 'failed'])
        failed = total - success
        
        # 统计诊断发现
        all_diagnoses = []
        for result in self.results:
            if result.get('diagnoses'):
                all_diagnoses.extend(result['diagnoses'])
        
        # 统计严重程度分布
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for diag in all_diagnoses:
            severity = diag.get('severity', 'low')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 统计最常见的发现
        findings_count = {}
        for diag in all_diagnoses:
            finding = f"{diag.get('organ', '')}_{diag.get('finding', '')}"
            findings_count[finding] = findings_count.get(finding, 0) + 1
        
        # 排序获取Top 5
        top_findings = sorted(findings_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary = {
            'total_files': total,
            'success_count': success,
            'failed_count': failed,
            'total_diagnoses': len(all_diagnoses),
            'severity_distribution': severity_counts,
            'top_findings': [{'finding': f, 'count': c} for f, c in top_findings],
            'processing_time': datetime.now().isoformat()
        }
        
        return summary
    
    def print_summary(self):
        """打印摘要报告"""
        summary = self.generate_summary_report()
        
        print("\n" + "="*60)
        print("批量处理摘要报告")
        print("="*60)
        print(f"总文件数: {summary['total_files']}")
        print(f"成功: {summary['success_count']}")
        print(f"失败: {summary['failed_count']}")
        print(f"总诊断发现数: {summary['total_diagnoses']}")
        print("\n严重程度分布:")
        print(f"  高危: {summary['severity_distribution']['high']}")
        print(f"  中危: {summary['severity_distribution']['medium']}")
        print(f"  低危: {summary['severity_distribution']['low']}")
        print("\nTop 5 常见发现:")
        for i, item in enumerate(summary['top_findings'], 1):
            print(f"  {i}. {item['finding']}: {item['count']} 例")
        print("="*60 + "\n")


class BatchUploadManager:
    """批量文件上传管理器"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """初始化上传管理器
        
        Args:
            temp_dir: 临时目录
        """
        import tempfile
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.uploaded_files = []
    
    def add_file(self, file_data, filename: str) -> str:
        """添加上传文件
        
        Args:
            file_data: 文件数据
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        file_path = os.path.join(self.temp_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        self.uploaded_files.append(file_path)
        return file_path
    
    def get_file_list(self) -> List[str]:
        """获取已上传文件列表
        
        Returns:
            list: 文件路径列表
        """
        return self.uploaded_files.copy()
    
    def clear(self):
        """清理临时文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.uploaded_files = []


# 测试代码
if __name__ == "__main__":
    from modules.inference_engine import MockInferenceEngine
    
    # 创建测试
    engine = MockInferenceEngine()
    processor = BatchProcessor(engine, max_workers=2)
    
    print("批量处理器初始化成功")
    print(f"最大worker数: {processor.max_workers}")
