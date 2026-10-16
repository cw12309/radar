"""
历史记录管理模块
使用SQLite数据库存储影像和诊断历史记录
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class HistoryManager:
    """诊断历史记录管理器"""
    
    def __init__(self, db_path: str = "radar_history.db"):
        """初始化历史记录管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建病例表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_time REAL,
                status TEXT DEFAULT 'pending',
                notes TEXT
            )
        ''')
        
        # 创建诊断结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                organ TEXT NOT NULL,
                finding TEXT NOT NULL,
                confidence REAL,
                severity TEXT,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases (id) ON DELETE CASCADE
            )
        ''')
        
        # 创建器官检测表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detected_organs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                organ_name TEXT NOT NULL,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases (id) ON DELETE CASCADE
            )
        ''')
        
        # 创建元数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                FOREIGN KEY (case_id) REFERENCES cases (id) ON DELETE CASCADE
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patient_id ON cases(patient_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_upload_time ON cases(upload_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_case_id ON diagnoses(case_id)')
        
        conn.commit()
        conn.close()
    
    def add_case(self,
                patient_id: str,
                file_name: str,
                file_path: Optional[str] = None,
                diagnosis_result: Optional[Dict] = None,
                metadata: Optional[Dict] = None) -> int:
        """添加新病例
        
        Args:
            patient_id: 患者ID
            file_name: 文件名
            file_path: 文件路径
            diagnosis_result: 诊断结果字典
            metadata: 元数据字典
            
        Returns:
            int: 新病例的ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 插入病例
        cursor.execute('''
            INSERT INTO cases (patient_id, file_name, file_path, status)
            VALUES (?, ?, ?, ?)
        ''', (patient_id, file_name, file_path, 'completed' if diagnosis_result else 'pending'))
        
        case_id = cursor.lastrowid
        
        # 插入诊断结果
        if diagnosis_result:
            # 插入器官检测
            organs = diagnosis_result.get('organs_detected', [])
            for organ in organs:
                cursor.execute('''
                    INSERT INTO detected_organs (case_id, organ_name)
                    VALUES (?, ?)
                ''', (case_id, organ))
            
            # 插入诊断发现
            diagnoses = diagnosis_result.get('diagnoses', [])
            for diag in diagnoses:
                cursor.execute('''
                    INSERT INTO diagnoses (case_id, organ, finding, confidence, severity)
                    VALUES (?, ?, ?, ?, ?)
                ''', (case_id, diag.get('organ'), diag.get('finding'), 
                     diag.get('confidence'), diag.get('severity')))
        
        # 插入元数据
        if metadata:
            for key, value in metadata.items():
                cursor.execute('''
                    INSERT INTO metadata (case_id, key, value)
                    VALUES (?, ?, ?)
                ''', (case_id, key, json.dumps(value) if isinstance(value, (dict, list)) else str(value)))
        
        conn.commit()
        conn.close()
        
        return case_id
    
    def get_case(self, case_id: int) -> Optional[Dict]:
        """获取病例详情
        
        Args:
            case_id: 病例ID
            
        Returns:
            dict: 病例详情字典
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 获取病例基本信息
        cursor.execute('SELECT * FROM cases WHERE id = ?', (case_id,))
        case_row = cursor.fetchone()
        
        if not case_row:
            conn.close()
            return None
        
        case = dict(case_row)
        
        # 获取器官检测
        cursor.execute('SELECT organ_name FROM detected_organs WHERE case_id = ?', (case_id,))
        case['organs_detected'] = [row['organ_name'] for row in cursor.fetchall()]
        
        # 获取诊断结果
        cursor.execute('SELECT * FROM diagnoses WHERE case_id = ?', (case_id,))
        case['diagnoses'] = [dict(row) for row in cursor.fetchall()]
        
        # 获取元数据
        cursor.execute('SELECT key, value FROM metadata WHERE case_id = ?', (case_id,))
        case['metadata'] = {row['key']: row['value'] for row in cursor.fetchall()}
        
        conn.close()
        return case
    
    def list_cases(self,
                  patient_id: Optional[str] = None,
                  limit: int = 50,
                  offset: int = 0,
                  order_by: str = 'upload_time DESC') -> List[Dict]:
        """列出病例列表
        
        Args:
            patient_id: 按患者ID筛选
            limit: 返回数量限制
            offset: 偏移量
            order_by: 排序字段
            
        Returns:
            list: 病例列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT * FROM cases'
        params = []
        
        if patient_id:
            query += ' WHERE patient_id = ?'
            params.append(patient_id)
        
        query += f' ORDER BY {order_by} LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        cases = [dict(row) for row in cursor.fetchall()]
        
        # 为每个病例添加诊断数量
        for case in cases:
            cursor.execute('SELECT COUNT(*) as count FROM diagnoses WHERE case_id = ?', (case['id'],))
            case['diagnoses_count'] = cursor.fetchone()['count']
        
        conn.close()
        return cases
    
    def search_cases(self,
                    keyword: Optional[str] = None,
                    finding: Optional[str] = None,
                    severity: Optional[str] = None,
                    date_from: Optional[str] = None,
                    date_to: Optional[str] = None) -> List[Dict]:
        """搜索病例
        
        Args:
            keyword: 关键词（搜索患者ID、文件名）
            finding: 诊断发现
            severity: 严重程度
            date_from: 开始日期
            date_to: 结束日期
            
        Returns:
            list: 匹配的病例列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT DISTINCT c.* FROM cases c'
        conditions = []
        params = []
        
        if finding or severity:
            query += ' LEFT JOIN diagnoses d ON c.id = d.case_id'
        
        if keyword:
            conditions.append('(c.patient_id LIKE ? OR c.file_name LIKE ?)')
            params.extend([f'%{keyword}%', f'%{keyword}%'])
        
        if finding:
            conditions.append('d.finding LIKE ?')
            params.append(f'%{finding}%')
        
        if severity:
            conditions.append('d.severity = ?')
            params.append(severity)
        
        if date_from:
            conditions.append('c.upload_time >= ?')
            params.append(date_from)
        
        if date_to:
            conditions.append('c.upload_time <= ?')
            params.append(date_to)
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        query += ' ORDER BY c.upload_time DESC'
        
        cursor.execute(query, params)
        cases = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return cases
    
    def delete_case(self, case_id: int) -> bool:
        """删除病例
        
        Args:
            case_id: 病例ID
            
        Returns:
            bool: 是否删除成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM cases WHERE id = ?', (case_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def get_statistics(self) -> Dict:
        """获取统计信息
        
        Returns:
            dict: 统计数据
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总病例数
        cursor.execute('SELECT COUNT(*) as count FROM cases')
        total_cases = cursor.fetchone()[0]
        
        # 总诊断数
        cursor.execute('SELECT COUNT(*) as count FROM diagnoses')
        total_diagnoses = cursor.fetchone()[0]
        
        # 严重程度分布
        cursor.execute('''
            SELECT severity, COUNT(*) as count 
            FROM diagnoses 
            GROUP BY severity
        ''')
        severity_dist = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 最常见的发现
        cursor.execute('''
            SELECT organ, finding, COUNT(*) as count 
            FROM diagnoses 
            GROUP BY organ, finding 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        top_findings = [
            {'organ': row[0], 'finding': row[1], 'count': row[2]}
            for row in cursor.fetchall()
        ]
        
        # 最近7天的活动
        cursor.execute('''
            SELECT DATE(upload_time) as date, COUNT(*) as count 
            FROM cases 
            WHERE upload_time >= datetime('now', '-7 days')
            GROUP BY DATE(upload_time)
            ORDER BY date
        ''')
        recent_activity = [
            {'date': row[0], 'count': row[1]}
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return {
            'total_cases': total_cases,
            'total_diagnoses': total_diagnoses,
            'severity_distribution': severity_dist,
            'top_findings': top_findings,
            'recent_activity': recent_activity
        }
    
    def export_to_json(self, output_path: str, case_ids: Optional[List[int]] = None):
        """导出数据为JSON
        
        Args:
            output_path: 输出文件路径
            case_ids: 要导出的病例ID列表，None表示全部
        """
        if case_ids:
            cases = [self.get_case(cid) for cid in case_ids]
            cases = [c for c in cases if c is not None]
        else:
            cases = self.list_cases(limit=10000)  # 导出所有
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cases, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"已导出 {len(cases)} 条记录到 {output_path}")


# 测试代码
if __name__ == "__main__":
    # 创建测试数据库
    manager = HistoryManager("test_history.db")
    
    # 测试添加病例
    test_result = {
        'organs_detected': ['肝', '肾', '脾'],
        'diagnoses': [
            {'organ': '肝', 'finding': '肝囊肿', 'confidence': 0.92, 'severity': 'low'}
        ]
    }
    
    case_id = manager.add_case(
        patient_id='TEST001',
        file_name='test.nii.gz',
        diagnosis_result=test_result
    )
    
    print(f"创建病例ID: {case_id}")
    
    # 测试获取病例
    case = manager.get_case(case_id)
    print(f"病例详情: {case}")
    
    # 测试统计
    stats = manager.get_statistics()
    print(f"统计信息: {stats}")
