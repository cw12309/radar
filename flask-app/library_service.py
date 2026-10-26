"""院内影像库服务

从「导出的快照目录」读取病例/检查/序列/报告索引，让 RADAR 能直接选用服务器上
已有的 DICOM 数据阅片诊断，而不必先下载到本地再用浏览器上传回去（腹部库单个
检查就有几百 MB，全库 47 GB）。

快照目录由 export_abdomen.py 生成，结构：
    <root>/cases_index.json        病例主索引（含分类、患者、检查号、DICOM 路径）
    <root>/reports_all.json        全部报告（findings/impression/clinical_history）
    <root>/dicom_index.json        检查级索引（扁平，无序列明细）
    <root>/cases/<二级>/<三级>/<病历号_姓名_id>/
        ├── case.json  reports.json  examinations.json
        └── dicom/dicom_index.json  ← 含每个检查的 series 明细，本模块读这个

设计要点：
  * 只读快照，不连 qinghua 的 PostgreSQL，不碰对方容器；
  * 库内影像零拷贝：file_id 经注册表映射到磁盘上的原始序列目录；
  * 序列适用性打分必须显式给出「不可用原因」——RADAR 只吃单一轴位 CT 序列，
    把 Scout / Dose Report / 冠状位重建喂进去不会报错，只会静默产出解剖
    错乱的诊断结果，这比直接失败更危险。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
from collections import defaultdict

import pydicom

# 库内 file_id 前缀。app.py 的 resolve_file_path 据此把 file_id 映射回真实路径，
# 其余路径（上传文件）仍按 UPLOAD_FOLDER 拼接。
LIB_PREFIX = 'lib_'


def _b64u(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode('utf-8')).decode('ascii').rstrip('=')


def _read_json(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ==================== 序列适用性判定 ====================

# 明确不能用于诊断的序列：定位像、剂量报告、三维重建/投影图。这些不是横断
# 断层图像，RADAR 拿到后不会报错，只会输出无意义的结果。
_RE_HARD_REJECT = re.compile(
    r'dose\s*report|dose\s*screen|scout|topogram|topo|localizer|loc\s*$'
    r'|\bvr\b|volume\s*render|\bmip\b|min\s*ip|max\s*ip|\b3d\b|\bssd\b'
    r'|screen\s*save|save\s*fluro|fluoro|derived|secondary\s*capture'
    r'|survey|scanogram|\bpos\b|positioning',
    re.I,
)
# 非横断位的重建序列（冠状位/矢状位/斜位 MPR）
_RE_NON_AXIAL = re.compile(
    r'\bcor\b|\bcoronal\b|\bsag\b|\bsagittal\b|\boblique\b|\bmpr\b|\breformat\b|\bcurved\b',
    re.I,
)
# 对比剂期相：RADAR 在 40 万例**增强**腹部 CT 上训练，平扫属于分布外输入，
# 现有 _assess_input_quality 会在诊断后给警告，这里前置到选择阶段提示。
_RE_PHASE_ARTERIAL = re.compile(r'\ba\+\b|arterial|动脉', re.I)
_RE_PHASE_VENOUS = re.compile(r'\bv\+\b|venous|portal|门脉|门静脉|静脉', re.I)
_RE_PHASE_DELAY = re.compile(r'\bd\+\b|\bp\+\b|delay|15\s*min|延迟', re.I)
_RE_PHASE_PLAIN = re.compile(r'\bc-\b|plain|non-?contrast|平扫|nc\b', re.I)

# 诊断可用的最少层数：RADAR 预处理会重采样到 1×1×5mm 并裁剪到 96 层，
# 层数过少（如 2 张的定位像）Z 轴信息不足以支撑器官级判断。
_MIN_INSTANCES = 20
# 层厚上限：超过 10mm 的厚层重建在 Z 轴上已丢失小病灶，与训练分布差异大。
_MAX_THICKNESS = 10.0


def _orientation_from_iop(iop):
    """由 ImageOrientationPatient 求层面法向，返回 (是否轴位, 法向 z 分量)。

    IOP 前三个数是行方向余弦、后三个是列方向余弦，二者叉乘即层面法向。
    轴位（横断位）时法向与人体 Z 轴平行，|n_z| ≈ 1；冠状位 n_z ≈ 0、
    矢状位 n_z ≈ 0，斜位介于两者之间。
    """
    if not iop or len(iop) < 6:
        return None, None
    try:
        vals = [float(v) for v in iop[:6]]
    except (TypeError, ValueError):
        return None, None
    r, c = vals[0:3], vals[3:6]
    nz = r[0] * c[1] - r[1] * c[0]
    return abs(nz) > 0.9, nz


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class SeriesAssessment:
    """单个序列的适用性评估结果。"""

    __slots__ = ('score', 'verdict', 'reasons', 'warnings', 'is_axial',
                 'orientation_known', 'thickness', 'instance_count', 'phase')

    def __init__(self):
        self.score = 0
        self.verdict = 'unsuitable'      # recommended | usable | unsuitable
        self.reasons = []                # 不可用的硬原因
        self.warnings = []               # 可用但需注意
        self.is_axial = None
        self.orientation_known = False
        self.thickness = None
        self.instance_count = 0
        self.phase = None

    def to_dict(self):
        return {
            'score': self.score,
            'verdict': self.verdict,
            'reasons': self.reasons,
            'warnings': self.warnings,
            'is_axial': self.is_axial,
            'orientation_known': self.orientation_known,
            'thickness': self.thickness,
            'instance_count': self.instance_count,
            'phase': self.phase,
        }


def assess_series(series, first_file=None):
    """评估一个序列能否用于 RADAR 诊断。

    series 为快照里的 series 记录（含 series_description / instance_count /
    modality / storage_path）；first_file 为该序列任一 DICOM 文件路径，用于
    读取 ImageOrientationPatient 与 SliceThickness（只读文件头，不解像素）。
    """
    a = SeriesAssessment()
    desc = (series.get('series_description') or '').strip()
    modality = (series.get('modality') or '').upper()
    count = int(series.get('instance_count') or series.get('image_count') or 0)
    a.instance_count = count

    # --- 硬拒：模态 ---
    if modality and modality != 'CT':
        a.reasons.append(f'RADAR 仅支持 CT，当前为 {modality}')
        a.score = -100

    # --- 硬拒：非断层图像（定位像/剂量报告/三维重建）---
    if desc and _RE_HARD_REJECT.search(desc):
        a.reasons.append('非断层序列（定位像/剂量报告/三维重建），无法用于诊断')
        a.score = -100

    # --- 轴位判定：优先用 DICOM 头的真实方向余弦 ---
    thickness = _to_float(series.get('slice_thickness'))
    if first_file and os.path.isfile(first_file):
        try:
            ds = pydicom.dcmread(first_file, stop_before_pixels=True)
            axial, nz = _orientation_from_iop(getattr(ds, 'ImageOrientationPatient', None))
            if axial is not None:
                a.orientation_known = True
                a.is_axial = axial
                if not axial:
                    a.reasons.append('非横断位（冠状/矢状/斜位重建）')
                    a.score = min(a.score, -100) if a.score < 0 else -100
            if thickness is None:
                thickness = _to_float(getattr(ds, 'SliceThickness', None))
        except Exception:
            pass
    # 头部拿不到方向时退回描述文本启发式
    if not a.orientation_known and desc and _RE_NON_AXIAL.search(desc):
        a.is_axial = False
        a.reasons.append('序列名提示为非横断位重建')
        a.score = -100

    a.thickness = thickness

    # --- 硬拒：层数过少 ---
    if count and count < _MIN_INSTANCES:
        a.reasons.append(f'仅 {count} 层，少于诊断所需的 {_MIN_INSTANCES} 层')
        if a.score >= 0:
            a.score = -80

    if a.score < 0:
        a.verdict = 'unsuitable'
        return a

    # --- 打分（越高越适合作为主诊断序列）---
    score = 40 if a.is_axial is not False else 10
    if count >= 60:
        score += 20
    elif count >= 40:
        score += 14
    else:
        score += 6

    if thickness is not None:
        if thickness <= 1.5:
            score += 12
            a.warnings.append(f'{thickness:g}mm 超薄层，层数多、载入稍慢')
        elif thickness <= 5.0:
            score += 15          # 与 RADAR 参考间距 1×1×5mm 最接近
        elif thickness <= _MAX_THICKNESS:
            score += 8
        else:
            score -= 10
            a.warnings.append(f'{thickness:g}mm 厚层，小病灶可能显示不清')

    # 期相识别：增强期相优先（训练分布），平扫给警告
    phase = None
    if desc:
        if _RE_PHASE_ARTERIAL.search(desc):
            phase = '动脉期'
        elif _RE_PHASE_DELAY.search(desc):
            phase = '延迟期'
        elif _RE_PHASE_VENOUS.search(desc):
            phase = '静脉期'
        elif _RE_PHASE_PLAIN.search(desc):
            phase = '平扫'
    a.phase = phase
    if phase == '平扫':
        score += 4
        a.warnings.append('平扫序列：RADAR 基于增强 CT 训练，平扫结果可靠性下降')
    elif phase in ('动脉期', '静脉期', '延迟期'):
        score += 18
    else:
        score += 8              # 期相不明，中性略偏正

    a.score = score
    a.verdict = 'recommended' if score >= 70 else 'usable'
    return a


def pick_first_dicom(series_dir):
    """返回序列目录里的第一个 DICOM 文件路径（用于只读文件头）。"""
    try:
        entries = sorted(os.scandir(series_dir), key=lambda e: e.name)
    except OSError:
        return None
    for entry in entries:
        if entry.is_file():
            return entry.path
    return None


# ==================== 影像库索引 ====================

class LibraryRoot:
    """单个快照目录的索引。"""

    def __init__(self, path, name=None):
        self.path = os.path.realpath(path)
        self.name = name or os.path.basename(self.path)
        self.cases = []                 # 顺序列表
        self.by_case_id = {}
        self.by_exam_id = {}
        self.by_series_id = {}
        self.reports_by_case = defaultdict(list)
        self.loaded_at = None
        self.error = None

    # ---------- 载入 ----------
    def load(self):
        manifests = _read_json(os.path.join(self.path, 'cases_index.json'))
        reports = _read_json(os.path.join(self.path, 'reports_all.json')) or []
        if manifests is None:
            self.error = f'{self.path}/cases_index.json 不存在或无法解析'
            return self

        self.reports_by_case = defaultdict(list)
        report_by_patient = defaultdict(list)
        for r in reports:
            if r.get('case_id'):
                self.reports_by_case[r['case_id']].append(r)
            if r.get('patient_id'):
                report_by_patient[r['patient_id']].append(r)

        self.cases = []
        self.by_case_id = {}
        self.by_exam_id = {}
        self.by_series_id = {}
        for m in manifests:
            case_id = m.get('case_id')
            if not case_id:
                continue
            # 报告并集：case_id 直接命中 + patient_id 命中（与导出脚本一致）
            merged = {r['id']: r for r in self.reports_by_case.get(case_id, [])}
            pid = m.get('patient_id')
            if pid:
                for r in report_by_patient.get(pid, []):
                    merged.setdefault(r['id'], r)
            case_reports = sorted(
                merged.values(),
                key=lambda r: (r.get('created_at') or '', r.get('report_type') or ''),
            )

            exams = _read_json(
                os.path.join(self.path, m.get('病例目录', ''), 'dicom', 'dicom_index.json')
            ) or []
            for e in exams:
                eid = e.get('examination_id')
                if eid:
                    self.by_exam_id[eid] = {'exam': e, 'case_id': case_id}
                for s in e.get('series') or []:
                    sid = s.get('id') or s.get('series_instance_uid')
                    if sid:
                        self.by_series_id[sid] = {
                            'series': s, 'exam': e, 'case_id': case_id,
                        }

            case = {
                'case_id': case_id,
                'patient_id': pid,
                'medical_record_no': m.get('病历号'),
                'patient_name': m.get('患者姓名'),
                'gender': m.get('性别'),
                'age': m.get('年龄'),
                'level1': m.get('一级分类'),
                'level2': m.get('二级分类'),
                'level3': m.get('三级分类'),
                'disease_name': m.get('病名(旧字段)') or m.get('选定诊断'),
                'selected_diagnosis': m.get('选定诊断'),
                'examination_part': m.get('检查部位'),
                'examination_method': m.get('检查方法'),
                'difficulty': m.get('难度'),
                'publication_status': m.get('发布状态'),
                'is_reviewed': m.get('已审核'),
                'ct_date': m.get('CT日期'),
                'mr_date': m.get('MR日期'),
                'case_dir': m.get('病例目录'),
                'exam_count': int(m.get('检查数') or 0),
                'modalities': m.get('检查模态') or '',
                'accession_numbers': m.get('检查号') or '',
                'dicom_bytes': int(m.get('DICOM总字节') or 0),
                'report_count': len(case_reports),
                'reports': case_reports,
                'examinations': exams,
            }
            self.cases.append(case)
            self.by_case_id[case_id] = case

        from datetime import datetime
        self.loaded_at = datetime.now().isoformat(timespec='seconds')
        return self

    # ---------- 查询 ----------
    def fingerprint(self):
        """快照指纹：用于判断是否需要重载（导出脚本重跑后自动生效）。"""
        key = os.path.join(self.path, 'cases_index.json')
        try:
            st = os.stat(key)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def tree(self):
        """三级分类树 + 各级病例计数。"""
        nodes = defaultdict(lambda: {'count': 0, 'dicom': 0})
        children = defaultdict(list)      # 父 key -> 子 key 顺序表
        for c in self.cases:
            l1 = c['level1'] or '未分类'
            l2 = c['level2'] or '未分类'
            l3 = c['level3'] or '未分类'
            for key, has_dicom in (((l1,), c['exam_count'] > 0),
                                   ((l1, l2), c['exam_count'] > 0),
                                   ((l1, l2, l3), c['exam_count'] > 0)):
                node = nodes[key]
                if node['count'] == 0:
                    parent = key[:-1]
                    if key not in children[parent]:
                        children[parent].append(key)
                node['count'] += 1
                if has_dicom:
                    node['dicom'] += 1

        def build(key, level):
            node = nodes[key]
            item = {
                'title': key[-1], 'key': '/'.join(key), 'level': level,
                'count': node['count'], 'dicom': node['dicom'],
            }
            kids = [build(child, level + 1) for child in sorted(children.get(key, []))]
            if kids:
                item['children'] = kids
            else:
                item['isLeaf'] = True
            return item

        return [build(k, 1) for k in sorted(children.get((), []))]

    def query_cases(self, category=None, keyword=None, modality=None,
                    has_dicom=None, has_report=None, status=None,
                    page=1, page_size=20):
        """按分类/关键词/模态/是否有影像或报告筛选病例，分页返回。"""
        rows = self.cases
        if category:
            parts = [p for p in category.split('/') if p]
            if len(parts) >= 1:
                rows = [c for c in rows if (c['level1'] or '未分类') == parts[0]]
            if len(parts) >= 2:
                rows = [c for c in rows if (c['level2'] or '未分类') == parts[1]]
            if len(parts) >= 3:
                rows = [c for c in rows if (c['level3'] or '未分类') == parts[2]]
        if keyword:
            kw = keyword.strip().lower()
            def hit(c):
                return kw in (c['medical_record_no'] or '').lower() \
                    or kw in (c['patient_name'] or '').lower() \
                    or kw in (c['accession_numbers'] or '').lower() \
                    or kw in (c['level3'] or '').lower() \
                    or kw in (c['disease_name'] or '').lower() \
                    or kw in (c['case_id'] or '').lower()
            rows = [c for c in rows if hit(c)]
        if modality:
            rows = [c for c in rows if modality.upper() in (c['modalities'] or '').upper().split(',')]
        if has_dicom is True:
            rows = [c for c in rows if c['exam_count'] > 0]
        elif has_dicom is False:
            rows = [c for c in rows if c['exam_count'] == 0]
        if has_report is True:
            rows = [c for c in rows if c['report_count'] > 0]
        elif has_report is False:
            rows = [c for c in rows if c['report_count'] == 0]
        if status:
            rows = [c for c in rows if (c['publication_status'] or '') == status]

        total = len(rows)
        page = max(1, int(page or 1))
        page_size = max(1, min(200, int(page_size or 20)))
        start = (page - 1) * page_size
        page_rows = rows[start:start + page_size]
        return {
            'total': total, 'page': page, 'page_size': page_size,
            'items': [self.case_summary(c) for c in page_rows],
        }

    @staticmethod
    def case_summary(c):
        """列表行数据（不含报告全文与序列明细，控制响应体积）。"""
        return {
            'case_id': c['case_id'],
            'medical_record_no': c['medical_record_no'],
            'patient_name': c['patient_name'],
            'gender': c['gender'],
            'age': c['age'],
            'level1': c['level1'], 'level2': c['level2'], 'level3': c['level3'],
            'disease_name': c['disease_name'],
            'examination_part': c['examination_part'],
            'examination_method': c['examination_method'],
            'difficulty': c['difficulty'],
            'publication_status': c['publication_status'],
            'is_reviewed': c['is_reviewed'],
            'exam_count': c['exam_count'],
            'modalities': c['modalities'],
            'accession_numbers': c['accession_numbers'],
            'dicom_gb': round(c['dicom_bytes'] / 1024 ** 3, 2),
            'report_count': c['report_count'],
            'has_dicom': c['exam_count'] > 0,
        }

    def case_detail(self, case_id):
        c = self.by_case_id.get(case_id)
        if c is None:
            return None
        exams = []
        for e in c['examinations']:
            series = [self.series_brief(s) for s in (e.get('series') or [])]
            exams.append({
                'examination_id': e.get('examination_id'),
                'modality': e.get('modality'),
                'accession_number': e.get('accession_number'),
                'study_instance_uid': e.get('study_instance_uid'),
                'study_description': e.get('study_description'),
                'body_part_examined': e.get('body_part_examined'),
                'examination_date': e.get('examination_date'),
                'series_count': e.get('series_count'),
                'image_count': e.get('image_count'),
                'size_gb': round(int(e.get('file_size_bytes') or 0) / 1024 ** 3, 2),
                'host_storage_path': e.get('host_storage_path'),
                'dicom_dir_exists': e.get('dicom_dir_exists'),
                'series': series,
            })
        reports = [
            {
                'id': r.get('id'),
                'report_type': r.get('report_type'),
                'match_status': r.get('match_status'),
                'source_accession_number': r.get('source_accession_number'),
                'source_modality': r.get('source_modality'),
                'source_exam_name': r.get('source_exam_name'),
                'clinical_history': r.get('clinical_history'),
                'findings': r.get('findings'),
                'impression': r.get('impression'),
                'report_content': r.get('report_content'),
                'created_at': r.get('created_at'),
            }
            for r in c['reports']
        ]
        detail = self.case_summary(c)
        detail.update({
            'patient_id': c['patient_id'],
            'case_dir': c['case_dir'],
            'ct_date': c['ct_date'], 'mr_date': c['mr_date'],
            'selected_diagnosis': c['selected_diagnosis'],
            'examinations': exams,
            'reports': reports,
        })
        return detail

    @staticmethod
    def series_brief(s):
        return {
            'series_id': s.get('id') or s.get('series_instance_uid'),
            'series_instance_uid': s.get('series_instance_uid'),
            'series_number': s.get('series_number'),
            'series_description': s.get('series_description'),
            'modality': s.get('modality'),
            'instance_count': s.get('instance_count') or s.get('image_count'),
            'storage_path': s.get('storage_path'),
        }

    def series_of_exam(self, examination_id):
        """列出某检查的全部序列，并逐个给出适用性评估与推荐排序。"""
        entry = self.by_exam_id.get(examination_id)
        if entry is None:
            return None
        exam = entry['exam']
        items = []
        for s in exam.get('series') or []:
            sid = s.get('id') or s.get('series_instance_uid')
            host_dir = self._series_host_dir(exam, s)
            assessment = assess_series(s, pick_first_dicom(host_dir) if host_dir else None)
            brief = self.series_brief(s)
            brief.update(assessment.to_dict())
            brief.update({
                'series_id': sid,
                'host_dir': host_dir,
                'dir_exists': bool(host_dir and os.path.isdir(host_dir)),
                'file_count_on_disk': _count_files(host_dir) if host_dir else 0,
            })
            items.append(brief)
        # 推荐序列排在最前，其余按打分降序、再按序列号
        items.sort(key=lambda x: (
            {'recommended': 0, 'usable': 1, 'unsuitable': 2}[x['verdict']],
            -x['score'], x['series_number'] or 0,
        ))
        best = next((i for i, x in enumerate(items) if x['verdict'] == 'recommended'), None)
        return {
            'examination_id': examination_id,
            'case_id': entry['case_id'],
            'study_instance_uid': exam.get('study_instance_uid'),
            'study_description': exam.get('study_description'),
            'modality': exam.get('modality'),
            'accession_number': exam.get('accession_number'),
            'host_storage_path': exam.get('host_storage_path'),
            'study_dir_exists': bool(exam.get('host_storage_path')
                                     and os.path.isdir(exam['host_storage_path'])),
            'series_count': len(items),
            'recommended_index': best,
            'series': items,
        }

    def _series_host_dir(self, exam, series):
        """把快照里的容器路径 /data/images/... 换算成本机真实路径。

        优先用 series.storage_path 相对 study 目录的后缀拼到 host_storage_path
        上——导出脚本记录的 container_storage_path 是容器视角，本机不一定有
        同名挂载点，而 host_storage_path 是 study 级的真实路径。
        """
        study_host = exam.get('host_storage_path')
        sp = series.get('storage_path') or ''
        if not study_host:
            return None
        if sp:
            # 取 series 路径的最后一段（= SeriesInstanceUID 目录名）
            tail = os.path.basename(sp.rstrip('/'))
            if tail:
                cand = os.path.join(study_host, tail)
                if os.path.isdir(cand):
                    return cand
        # 退化：直接按 SeriesInstanceUID 找子目录
        uid = series.get('series_instance_uid')
        if uid:
            cand = os.path.join(study_host, uid)
            if os.path.isdir(cand):
                return cand
        return None

    def find_series(self, series_id):
        return self.by_series_id.get(series_id)


def _count_files(directory):
    try:
        with os.scandir(directory) as it:
            return sum(1 for e in it if e.is_file())
    except OSError:
        return 0


# ==================== 全局注册表 ====================

class LibraryService:
    """管理多个快照根目录 + 库内 file_id 注册表。

    注册表把 lib_<hash> 映射到磁盘上的真实路径。之所以用不透明哈希而不是把
    路径直接编码进 file_id：file_id 会被拼进 URL 与 radar_infer/<file_id>
    输出目录，含 '/' 或过长都会出问题；哈希是单一合法路径段，且天然阻止
    通过 file_id 越权访问任意目录（只有登记过的序列能用）。
    """

    def __init__(self):
        self._roots = {}
        self._order = []
        self._lock = threading.RLock()
        self._registry = {}          # file_id -> {series_dir, study_dir, meta...}
        self._studies = {}           # study_uid -> study_dir
        self._series_cache = {}      # examination_id -> series_of_exam 结果
        self._registry_path = None

    # ---------- 配置 ----------
    def configure(self, roots, registry_path):
        """按 settings 里的 library_roots 装载快照目录（幂等，可热更新）。"""
        self._registry_path = registry_path
        self._load_registry()
        wanted = []
        for item in roots or []:
            path = item.get('path') if isinstance(item, dict) else item
            name = item.get('name') if isinstance(item, dict) else None
            if path:
                wanted.append((os.path.realpath(path), name))
        with self._lock:
            for path, name in wanted:
                if path not in self._roots:
                    self._roots[path] = LibraryRoot(path, name).load()
                    self._order.append(path)
            # 已移除的根：从索引里摘掉，但保留注册表条目（历史诊断仍可回看）
            for path in [p for p in self._order if p not in {w[0] for w in wanted}]:
                self._roots.pop(path, None)
                self._order.remove(path)

    def refresh(self):
        """快照被重新导出后重载（按 mtime+size 指纹判断，避免无谓重扫）。"""
        with self._lock:
            for path in self._order:
                root = self._roots.get(path)
                if root is None:
                    continue
                if root.fingerprint() != getattr(root, '_fp', None):
                    root.load()
                    root._fp = root.fingerprint()
                    self._series_cache.clear()

    def status(self):
        with self._lock:
            return [
                {
                    'name': r.name, 'path': r.path, 'error': r.error,
                    'cases': len(r.cases),
                    'examinations': len(r.by_exam_id),
                    'series': len(r.by_series_id),
                    'loaded_at': r.loaded_at,
                }
                for r in (self._roots[p] for p in self._order)
            ]

    # ---------- 查询代理 ----------
    def _iter_roots(self):
        with self._lock:
            return [self._roots[p] for p in self._order]

    def tree(self):
        out = []
        for r in self._iter_roots():
            if r.error:
                continue
            out.append({'root': r.name, 'path': r.path, 'tree': r.tree()})
        return out

    def query_cases(self, root=None, **kwargs):
        roots = self._iter_roots()
        if root:
            roots = [r for r in roots if r.name == root or r.path == root]
        if not roots:
            return {'total': 0, 'items': [], 'page': 1, 'page_size': 20}
        if len(roots) == 1:
            return roots[0].query_cases(**kwargs)
        # 多根：合并后自行分页
        merged = []
        for r in roots:
            res = r.query_cases(page=1, page_size=100000, **kwargs)
            for it in res['items']:
                it['root'] = r.name
            merged.extend(res['items'])
        page = max(1, int(kwargs.get('page') or 1))
        size = max(1, min(200, int(kwargs.get('page_size') or 20)))
        start = (page - 1) * size
        return {'total': len(merged), 'page': page, 'page_size': size,
                'items': merged[start:start + size]}

    def case_detail(self, case_id):
        for r in self._iter_roots():
            d = r.case_detail(case_id)
            if d is not None:
                d['root'] = r.name
                return d
        return None

    def series_of_exam(self, examination_id, force=False):
        if not force:
            with self._lock:
                cached = self._series_cache.get(examination_id)
            if cached is not None:
                return cached
        for r in self._iter_roots():
            res = r.series_of_exam(examination_id)
            if res is not None:
                with self._lock:
                    self._series_cache[examination_id] = res
                return res
        return None

    def find_series(self, series_id):
        for r in self._iter_roots():
            hit = r.find_series(series_id)
            if hit is not None:
                return hit, r
        return None, None

    # ---------- 注册表：库内 file_id ↔ 真实路径 ----------
    def _load_registry(self):
        if not self._registry_path or not os.path.isfile(self._registry_path):
            return
        data = _read_json(self._registry_path) or {}
        self._registry = data.get('files') or {}
        self._studies = data.get('studies') or {}

    def _save_registry(self):
        if not self._registry_path:
            return
        try:
            os.makedirs(os.path.dirname(self._registry_path), exist_ok=True)
            with open(self._registry_path, 'w', encoding='utf-8') as fh:
                json.dump({'files': self._registry, 'studies': self._studies},
                          fh, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def register_series(self, series_dir, study_dir, meta):
        """把一个库内序列目录注册为 file_id，返回 file_id。

        同一路径重复注册返回同一 file_id，这样对同一序列反复诊断会复用
        radar_infer/<file_id> 下的中间产物（NIfTI、overlay），不必重算。
        """
        series_dir = os.path.realpath(series_dir)
        study_dir = os.path.realpath(study_dir) if study_dir else None
        token = LIB_PREFIX + hashlib.sha1(
            f'{series_dir}|{study_dir}'.encode('utf-8')).hexdigest()[:16]
        with self._lock:
            self._registry[token] = {
                'series_dir': series_dir,
                'study_dir': study_dir,
                **meta,
            }
            study_uid = meta.get('study_instance_uid')
            if study_uid and study_dir:
                self._studies[study_uid] = study_dir
            self._save_registry()
        return token

    def resolve(self, file_id):
        """file_id → 真实路径；非库内 id 返回 None。"""
        if not isinstance(file_id, str) or not file_id.startswith(LIB_PREFIX):
            return None
        with self._lock:
            entry = self._registry.get(file_id)
        if not entry:
            return None
        path = entry['series_dir']
        # 防御：注册表被手工改坏时不要逃逸到任意路径
        if not os.path.exists(path):
            return None
        return path

    def registry_entry(self, file_id):
        with self._lock:
            return dict(self._registry.get(file_id) or {})

    def study_dir(self, study_uid):
        """OHIF 用：study_uid → 原始 study 目录（含全部序列子目录）。"""
        with self._lock:
            path = self._studies.get(study_uid)
        if path and os.path.isdir(path):
            return path
        return None

    def registered_study_uids(self):
        with self._lock:
            return list(self._studies.keys())

    # ---------- 外部数据的服务器侧导入 ----------
    def scan_directory(self, root_dir, max_depth=4, max_series=200):
        """扫描任意目录，按 SeriesInstanceUID 归组，返回可导入的序列清单。

        用于「数据已经在服务器上、但不在影像库快照里」的场景：不必先下载到
        本地再经浏览器传回来。只读每个序列的首帧文件头，逐个给出 RADAR
        适用性评估。递归深度受 max_depth 限制，避免误扫整个磁盘。
        """
        root_dir = os.path.realpath(root_dir)
        if not os.path.isdir(root_dir):
            return {'root': root_dir, 'groups': [], 'total_files': 0}

        groups = {}       # series_uid -> {study_uid, dir, count, first_file, desc}
        total = 0
        base_depth = root_dir.rstrip(os.sep).count(os.sep)

        def walk(current):
            nonlocal total
            if len(groups) >= max_series:
                return
            try:
                entries = list(os.scandir(current))
            except OSError:
                return
            depth = current.rstrip(os.sep).count(os.sep) - base_depth
            subdirs = [e for e in entries if e.is_dir(follow_symlinks=True)]
            files = [e for e in entries if e.is_file()]
            dcm_here = [e for e in files if e.name.lower().endswith('.dcm')]

            if dcm_here:
                # 该目录本身就是（部分）序列，读首帧确定归组
                first = sorted(dcm_here, key=lambda e: e.name)[0]
                meta = _read_series_header(first.path)
                uid = meta.get('series_uid') or current
                total += len(dcm_here)
                g = groups.setdefault(uid, {
                    'series_instance_uid': meta.get('series_uid'),
                    'study_instance_uid': meta.get('study_uid'),
                    'series_description': meta.get('series_description'),
                    'series_number': meta.get('series_number'),
                    'modality': meta.get('modality'),
                    'slice_thickness': meta.get('slice_thickness'),
                    'dir': current,
                    'instance_count': 0,
                    'first_file': first.path,
                })
                g['instance_count'] += len(dcm_here)
            elif depth < max_depth:
                for e in subdirs:
                    walk(e.path)

        walk(root_dir)

        items = []
        for uid, g in list(groups.items())[:max_series]:
            assessment = assess_series(g, g.get('first_file'))
            row = {
                'series_id': uid,
                'series_instance_uid': g.get('series_instance_uid'),
                'study_instance_uid': g.get('study_instance_uid'),
                'series_description': g.get('series_description'),
                'series_number': g.get('series_number'),
                'modality': g.get('modality'),
                'series_dir': g['dir'],
                'instance_count': g['instance_count'],
            }
            row.update(assessment.to_dict())
            items.append(row)
        items.sort(key=lambda x: (
            {'recommended': 0, 'usable': 1, 'unsuitable': 2}[x['verdict']],
            -x['score'], x['series_number'] or 0,
        ))
        best = next((i for i, x in enumerate(items) if x['verdict'] == 'recommended'), None)
        return {
            'root': root_dir, 'total_files': total,
            'series_count': len(items), 'recommended_index': best,
            'series': items,
        }


def _read_series_header(filepath):
    """只读文件头，抽取归组与打分所需的最小字段。"""
    out = {}
    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        out['series_uid'] = str(getattr(ds, 'SeriesInstanceUID', '') or '')
        out['study_uid'] = str(getattr(ds, 'StudyInstanceUID', '') or '')
        out['series_description'] = str(getattr(ds, 'SeriesDescription', '') or '') or None
        out['series_number'] = getattr(ds, 'SeriesNumber', None)
        out['modality'] = str(getattr(ds, 'Modality', '') or '') or None
        thickness = _to_float(getattr(ds, 'SliceThickness', None))
        if thickness is not None:
            out['slice_thickness'] = thickness
        axial, _nz = _orientation_from_iop(getattr(ds, 'ImageOrientationPatient', None))
        if axial is not None:
            out['is_axial'] = axial
    except Exception:
        pass
    return out


# 模块级单例
_service = LibraryService()


def get_library() -> LibraryService:
    return _service
