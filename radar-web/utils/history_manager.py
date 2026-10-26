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

        self._migrate(cursor)

        conn.commit()
        conn.close()

    # cases 表的完整列。列表查询只取 SUMMARY_COLUMNS（不含两个大 JSON 字段），
    # 避免每次翻页都把全部诊断结果读进内存再丢掉。
    SUMMARY_COLUMNS = (
        'id', 'patient_id', 'file_name', 'file_path', 'upload_time',
        'processing_time', 'status', 'notes', 'file_id', 'source',
        'risk_level', 'summary',
    )

    def _migrate(self, cursor):
        """给已存在的 cases 表补列（SQLite 无 IF NOT EXISTS，需先查再补）。

        历史记录库是长期累积的，直接改 CREATE TABLE 对老库无效；老记录的新列
        会是 NULL，读取时按缺省处理即可。
        """
        existing = {row[1] for row in cursor.execute('PRAGMA table_info(cases)')}
        # diagnosis_json：完整诊断结果（explanation/top_scores/warnings/metadata/
        #   overlay 元数据等），用于事后回看、复现与重新生成 PDF 报告。
        # summary：小体积摘要，列表页直接展示，不必解析完整 JSON。
        # library_json：院内影像库来源信息（病例/检查/序列 + 原始报告全文），
        #   使历史记录能追溯回影像库并再次做「AI vs 人工报告」对照。
        new_columns = (
            ('file_id', 'TEXT'),
            # 不给 DEFAULT：升级前的老记录来源无从判定，宁可留 NULL 由前端显示
            # 「来源未记录」，也不要一律兜底成 'upload' 造成误标。
            ('source', 'TEXT'),
            ('risk_level', 'TEXT'),
            ('summary', 'TEXT'),
            ('diagnosis_json', 'TEXT'),
            ('library_json', 'TEXT'),
        )
        for name, decl in new_columns:
            if name not in existing:
                cursor.execute(f'ALTER TABLE cases ADD COLUMN {name} {decl}')
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_risk_level ON cases(risk_level)')

        # 老记录没有 summary，列表页会一律显示「未发现达到阈值的异常」。
        # 用 diagnoses 关系表回填，保证升级后历史结论仍然正确。
        if 'summary' not in existing:
            self._backfill_summaries(cursor)

    def _backfill_summaries(self, cursor):
        """给升级前的老记录补摘要（仅能从 diagnoses 表还原结论部分）。"""
        cursor.execute('SELECT id, processing_time FROM cases')
        rows = cursor.fetchall()

        cursor.execute('''
            SELECT case_id, organ, finding, confidence, severity
            FROM diagnoses ORDER BY case_id, confidence DESC
        ''')
        by_case: Dict[int, List[Dict]] = {}
        for cid, organ, finding, confidence, severity in cursor.fetchall():
            by_case.setdefault(cid, []).append({
                'organ': organ, 'finding': finding,
                'confidence': confidence, 'severity': severity,
            })

        rank = {'high': 3, 'medium': 2, 'low': 1}
        for case_id, processing_time in rows:
            diagnoses = by_case.get(case_id, [])
            level = max((d['severity'] for d in diagnoses if d['severity'] in rank),
                        key=lambda s: rank[s], default=None)
            summary = {
                'risk_level': level or 'none',
                'findings_count': len(diagnoses),
                'diagnosis': '、'.join(
                    f"{d['organ']}-{d['finding']}" for d in diagnoses
                ) or '未发现达到阈值的异常',
                'diagnoses': diagnoses,
                'processing_time': processing_time,
                'legacy': True,  # 完整结果（explanation/warnings 等）升级前未留存
            }
            cursor.execute(
                'UPDATE cases SET summary = ?, risk_level = ? WHERE id = ?',
                (json.dumps(summary, ensure_ascii=False), summary['risk_level'], case_id))

    def add_case(self,
                patient_id: str,
                file_name: str,
                file_path: Optional[str] = None,
                diagnosis_result: Optional[Dict] = None,
                metadata: Optional[Dict] = None,
                file_id: Optional[str] = None,
                source: str = 'upload',
                library_context: Optional[Dict] = None) -> int:
        """添加新病例

        Args:
            patient_id: 患者ID
            file_name: 文件名（展示用）
            file_path: 文件路径
            diagnosis_result: 完整诊断结果字典，原样存进 diagnosis_json，
                保证事后能回看 explanation / top_scores / warnings 并重新出报告
            metadata: 元数据字典
            file_id: 影像标识（库内导入为 lib_<hash>，可据此复原 overlay/切片）
            source: 来源，'library'（院内影像库）或 'upload'（外部导入）
            library_context: 影像库溯源信息 + 原始报告，存进 library_json

        Returns:
            int: 新病例的ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 完整结果与摘要分开落库：摘要供列表页快速展示，完整结果按需读取
        full = self._enrich_result(diagnosis_result, metadata)
        summary = self._build_summary(diagnosis_result or {})

        # 插入病例
        cursor.execute('''
            INSERT INTO cases (patient_id, file_name, file_path, status,
                               file_id, source, risk_level, summary,
                               diagnosis_json, library_json, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (patient_id, file_name, file_path,
              'completed' if diagnosis_result else 'pending',
              file_id, source,
              summary['risk_level'],
              json.dumps(summary, ensure_ascii=False),
              json.dumps(full, ensure_ascii=False, default=str) if full else None,
              json.dumps(library_context, ensure_ascii=False, default=str)
              if library_context else None,
              summary.get('processing_time')))

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

    @staticmethod
    def _enrich_result(diagnosis_result, metadata):
        """把 metadata 并入完整结果，得到可独立复现报告的快照。"""
        if not diagnosis_result:
            return None
        full = dict(diagnosis_result)
        if metadata:
            full.setdefault('metadata', {}).update(metadata)
        return full

    @staticmethod
    def _build_summary(diagnosis_result: Dict) -> Dict:
        """从完整结果提炼小体积摘要（列表页 / 报告抬头直接用）。

        只保留结论性字段，不含 explanation 长文本与逐器官 top_scores，
        单条摘要稳定在几百字节量级。
        """
        diagnoses = diagnosis_result.get('diagnoses') or []
        order = {'high': 3, 'medium': 2, 'low': 1}
        risk = 'none'
        if diagnoses:
            risk = max((d.get('severity') or 'low' for d in diagnoses),
                       key=lambda s: order.get(s, 0))
        explanation = diagnosis_result.get('explanation') or {}
        top = diagnosis_result.get('organs_detected') or []
        return {
            'engine': diagnosis_result.get('engine'),
            'patient_id': diagnosis_result.get('patient_id'),
            'risk_level': risk,
            'findings_count': len(diagnoses),
            # 结论串：与旧版 API 的 diagnosis.diagnosis 字段保持一致的展示口径
            'diagnosis': '、'.join(
                f"{d.get('organ')}-{d.get('finding')}" for d in diagnoses
            ) or '未发现达到阈值的异常',
            'diagnoses': [
                {
                    'organ': d.get('organ'),
                    'finding': d.get('finding'),
                    'confidence': d.get('confidence'),
                    'severity': d.get('severity'),
                }
                for d in diagnoses
            ],
            'organs_detected_count': len(top),
            'warnings': diagnosis_result.get('warnings') or [],
            'processing_time': explanation.get('processing_time_sec'),
        }
    
    def get_case(self, case_id: int) -> Optional[Dict]:
        """获取病例详情

        Args:
            case_id: 病例ID

        Returns:
            dict: 病例详情字典，含解析后的 summary / diagnosis（完整结果）
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
        # 完整结果与摘要以 JSON 文本落库，读出来解析成对象
        case['summary'] = self._loads(case.get('summary'), {})
        case['diagnosis'] = self._loads(case.get('diagnosis_json'), None)
        case['library_context'] = self._loads(case.get('library_json'), None)

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

    def get_full_diagnosis(self, case_id: int) -> Optional[Dict]:
        """获取单条病例的完整诊断结果（用于详情弹窗 / 重新生成报告）。

        只取需要的大字段，避免连带整表加载。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT diagnosis_json, library_json, summary, file_id, source, patient_id, file_name '
            'FROM cases WHERE id = ?',
            (case_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return {
            'diagnosis': self._loads(row['diagnosis_json'], None),
            'library_context': self._loads(row['library_json'], None),
            'summary': self._loads(row['summary'], {}),
            # 重新出 PDF 报告需要 file_id 复原影像证据截图
            'file_id': row['file_id'],
            'source': row['source'],
            'patient_id': row['patient_id'],
            'file_name': row['file_name'],
        }

    @staticmethod
    def _loads(text, default):
        if not text:
            return default
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return default

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
            list: 病例列表（仅摘要列，不含大 JSON 字段）
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 只 SELECT 摘要列，跳过 diagnosis_json / library_json 两个大字段
        cols = ', '.join(self.SUMMARY_COLUMNS)
        query = f'SELECT {cols} FROM cases'
        params: List = []

        if patient_id:
            query += ' WHERE patient_id = ?'
            params.append(patient_id)

        query += f' ORDER BY {order_by} LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        cursor.execute(query, params)
        cases = [dict(row) for row in cursor.fetchall()]

        # 为每个病例附加解析好的摘要（含 diagnoses 明细），列表页无需再回查
        case_ids = [c['id'] for c in cases]
        counts = {}
        if case_ids:
            placeholders = ','.join('?' * len(case_ids))
            cursor.execute(
                f'SELECT case_id, COUNT(*) c FROM diagnoses WHERE case_id IN ({placeholders}) GROUP BY case_id',
                case_ids)
            counts = {row['case_id']: row['c'] for row in cursor.fetchall()}

        for case in cases:
            case['summary'] = self._loads(case.get('summary'), {})
            case['diagnoses_count'] = counts.get(case['id'], 0)

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

        # 同 list_cases：只取摘要列，不把完整诊断结果带进搜索结果
        cols = ', '.join(f'c.{name}' for name in self.SUMMARY_COLUMNS)
        query = f'SELECT DISTINCT {cols} FROM cases c'
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
        for case in cases:
            case['summary'] = self._loads(case.get('summary'), {})

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
