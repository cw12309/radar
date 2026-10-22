# -*- coding: utf-8 -*-
"""PDF报告生成模块

生成含患者信息、影像证据与诊断发现的专业中文PDF报告。

历史问题（本版修复）：
  1. 旧版直接使用 reportlab 默认 Helvetica 字体，无中文字形，生成的
     PDF 中文全部乱码；现启动时注册 Windows 系统中文字体（微软雅黑，
     回退黑体/宋体），全部失败时退回 reportlab 内置 CID 字体
     STSong-Light，保证中文永远可渲染。
  2. 旧版 charts 分支误用 plotly API（fig.write_image），传入
     matplotlib figure 时抛异常；现按 matplotlib 处理。
  3. 旧版报告无影像证据；现支持 evidence_images 参数嵌入 CT 截图
     （分割叠加/注意力热力图），做到"报告有图、结论有据"。
"""
import io
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, KeepTogether, HRFlowable, CondPageBreak,
)

# ---- 版式常量 ----
PRIMARY = colors.HexColor('#0f2d52')      # 报告主色（深蓝）
GRAY_BG = colors.HexColor('#f2f5f8')      # 表格隔行/标签底色
BORDER = colors.HexColor('#c8d2dc')       # 表格边线
TEXT_MAIN = colors.HexColor('#1c2733')    # 正文
TEXT_SUB = colors.HexColor('#5a6b7d')     # 次要文字
SEV_COLORS = {
    'high': colors.HexColor('#e74c3c'),
    'medium': colors.HexColor('#f39c12'),
    'low': colors.HexColor('#27ae60'),
}
SEV_LABELS = {'high': '高风险', 'medium': '中风险', 'low': '低风险'}
SEV_HEX = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#27ae60'}
SEV_ROW_BG = {
    'high': colors.HexColor('#fdecea'),
    'medium': colors.HexColor('#fef5e7'),
    'low': colors.HexColor('#eafaf1'),
}
CONTENT_WIDTH = 17 * cm  # A4 宽 21cm - 左右边距各 2cm

# ---- 标准版式模板（版本化）----
# 章节间距/标题间距/行距统一定义于此，报告各章节生成时一律引用本模板，
# 调整排版只改这里；generate_report(layout=...) 可按次针对性覆盖个别键。
REPORT_LAYOUT = {
    'version': 2,
    'section_gap': 0.55 * cm,          # 章节结束后与下一章节标题间的留白
    'section_title_before': 0,         # 章节标题前额外留白（间隔由上一章 section_gap 提供）
    'section_title_after': 0.3 * cm,   # 章节标题下划线与正文间的留白
    'body_leading': 17,                # 正文行距
    'header_gap': 0.35 * cm,           # 报告头与首章节间的留白
}

# 患者/检查信息展示顺序（键与 app.py extract_dicom_metadata 输出一致）
_META_FIELDS = [
    ('patient_name', '患者姓名'),
    ('patient_id', '患者ID'),
    ('patient_sex', '性别'),
    ('patient_age', '年龄'),
    ('patient_weight', '体重(kg)'),
    ('modality', '检查类型'),
    ('body_part', '检查部位'),
    ('study_date', '检查日期'),
    ('study_time', '检查时间'),
    ('study_description', '检查描述'),
    ('series_description', '序列描述'),
    ('manufacturer', '设备厂商'),
    ('institution', '检查机构'),
    ('slice_thickness', '层厚(mm)'),
    ('num_instances', '序列层数'),
    ('image_shape', '影像尺寸'),
]


def _register_chinese_fonts():
    """注册中文字体，返回 (常规体, 粗体) 的 reportlab 字体名。

    优先微软雅黑（现代观感），回退黑体/宋体；TrueType 全部失败时用
    reportlab 内置 CID 字体 STSong-Light（无需字体文件，任何环境可用）。
    """
    candidates = [
        ('RADAR-YH', 'RADAR-YH-Bold',
         'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyhbd.ttc'),
        ('RADAR-Hei', 'RADAR-Hei',
         'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/simhei.ttf'),
        ('RADAR-Song', 'RADAR-Song',
         'C:/Windows/Fonts/simsun.ttc', 'C:/Windows/Fonts/simsun.ttc'),
    ]
    for regular, bold, path, bold_path in candidates:
        try:
            pdfmetrics.registerFont(TTFont(regular, path, subfontIndex=0))
            if bold_path != path:
                pdfmetrics.registerFont(TTFont(bold, bold_path, subfontIndex=0))
            else:
                bold = regular
            return regular, bold
        except Exception:
            continue
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    return 'STSong-Light', 'STSong-Light'


def overall_severity(diagnoses):
    """总体风险分级：取全部阳性项最高严重度；无阳性项返回 None。"""
    if any(d.get('severity') == 'high' for d in diagnoses):
        return 'high'
    if any(d.get('severity') == 'medium' for d in diagnoses):
        return 'medium'
    if diagnoses:
        return 'low'
    return None


class PDFReportGenerator:
    """PDF医学影像诊断报告生成器（专业中文排版）。"""

    def __init__(self, layout: Optional[Dict] = None):
        self.font_name, self.font_bold = _register_chinese_fonts()
        self.layout = dict(REPORT_LAYOUT)
        if layout:
            self.layout.update(layout)
        self.styles = self._build_styles()

    # ------------------------------------------------------------------ #
    # 样式
    # ------------------------------------------------------------------ #
    def _build_styles(self) -> Dict[str, ParagraphStyle]:
        def ps(name, **kw):
            base = dict(fontName=self.font_name, fontSize=10, leading=15,
                        textColor=TEXT_MAIN)
            base.update(kw)
            return ParagraphStyle(name, **base)

        return {
            'title': ps('title', fontName=self.font_bold, fontSize=20, leading=27,
                        alignment=TA_CENTER, textColor=PRIMARY),
            'subtitle': ps('subtitle', fontSize=9.5, leading=13,
                           alignment=TA_CENTER, textColor=TEXT_SUB),
            'section': ps('section', fontName=self.font_bold, fontSize=13, leading=18,
                          spaceBefore=self.layout['section_title_before'],
                          spaceAfter=2, textColor=PRIMARY),
            'body': ps('body', fontSize=10, leading=self.layout['body_leading']),
            'small': ps('small', fontSize=8.5, leading=12.5, textColor=TEXT_SUB),
            'cell': ps('cell', fontSize=9, leading=13),
            'cellBold': ps('cellBold', fontName=self.font_bold, fontSize=9, leading=13),
            'cellLabel': ps('cellLabel', fontSize=9, leading=13, textColor=TEXT_SUB),
            'cellHead': ps('cellHead', fontName=self.font_bold, fontSize=9.5,
                           leading=13, textColor=colors.white),
            'caption': ps('caption', fontSize=8.5, leading=12,
                          alignment=TA_CENTER, textColor=TEXT_SUB),
            'summaryValue': ps('summaryValue', fontName=self.font_bold, fontSize=15,
                               leading=20, alignment=TA_CENTER),
            'summaryLabel': ps('summaryLabel', fontSize=8, leading=11,
                               alignment=TA_CENTER, textColor=TEXT_SUB),
        }

    def _section(self, text) -> List:
        """章节标题 + 主色下划线。"""
        return [
            Paragraph(text, self.styles['section']),
            HRFlowable(width='100%', thickness=1.1, color=PRIMARY,
                       spaceBefore=2,
                       spaceAfter=self.layout['section_title_after']),
        ]

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def generate_report(self,
                        output_path: str,
                        diagnosis_result: Dict,
                        image_slices: Optional[Dict] = None,
                        charts: Optional[Dict] = None,
                        metadata: Optional[Dict] = None,
                        evidence_images: Optional[List[Dict]] = None,
                        layout: Optional[Dict] = None) -> str:
        """生成完整PDF报告。

        Args:
            output_path: 输出PDF文件路径
            diagnosis_result: 诊断结果字典（diagnoses/explanation/organs_detected 等）
            image_slices: 兼容旧接口，{'axial'/'sagittal'/'coronal': ndarray}
            charts: 兼容旧接口，{name: matplotlib.figure.Figure}
            metadata: 患者/检查元数据（app.py extract_dicom_metadata 输出）
            evidence_images: 证据截图列表 [{'title': str, 'png': bytes}, ...]
            layout: 针对本次生成的版式覆盖（键见 REPORT_LAYOUT），
                未提供的键沿用标准模板

        Returns:
            str: 生成的PDF文件路径
        """
        # 按次覆盖版式：生成结束后还原，避免污染单例实例的后续报告
        saved = None
        if layout:
            saved = (dict(self.layout), self.styles)
            self.layout.update(layout)
            self.styles = self._build_styles()
        try:
            return self._build_report(
                output_path, diagnosis_result, image_slices, charts,
                metadata, evidence_images)
        finally:
            if saved:
                self.layout, self.styles = saved

    def _build_report(self,
                      output_path: str,
                      diagnosis_result: Dict,
                      image_slices: Optional[Dict],
                      charts: Optional[Dict],
                      metadata: Optional[Dict],
                      evidence_images: Optional[List[Dict]]) -> str:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=1.9 * cm,
            title='RADAR智能影像诊断报告',
            author='RADAR智能影像诊断平台',
        )

        sections = [
            self._create_patient_section(diagnosis_result, metadata or {}),
            self._create_summary_section(diagnosis_result),
            self._create_diagnosis_section(diagnosis_result),
        ]
        if evidence_images:
            sections.append(self._create_evidence_section(evidence_images))
        if image_slices:
            sections.append(self._create_image_section(image_slices))
        if charts:
            sections.append(self._create_statistics_section(charts))
        sections.append(self._create_explanation_section(diagnosis_result))
        sections.append(self._create_conclusion_section(diagnosis_result))

        # 章节间隔统一由版式模板控制：报告头后接 header_gap，其后每章之间
        # 接 section_gap（空章节跳过，不产生多余留白）
        story = []
        story.extend(self._create_header(diagnosis_result))
        pending_gap = self.layout['header_gap']
        for sec in sections:
            if not sec:
                continue
            # 页底剩余不足 2.5cm 时先换页，避免章节标题孤悬页底与正文分离
            story.append(CondPageBreak(2.5 * cm))
            story.append(Spacer(1, pending_gap))
            story.extend(sec)
            pending_gap = self.layout['section_gap']
        story.append(Spacer(1, pending_gap))
        story.extend(self._create_footer_section())

        doc.build(story, onFirstPage=self._draw_page_frame,
                  onLaterPages=self._draw_page_frame)
        return output_path

    # ------------------------------------------------------------------ #
    # 页眉页脚（canvas 绘制）
    # ------------------------------------------------------------------ #
    def _draw_page_frame(self, canvas, doc):
        canvas.saveState()
        # 顶部主色横幅
        canvas.setFillColor(PRIMARY)
        canvas.rect(0, A4[1] - 1.15 * cm, A4[0], 1.15 * cm, stroke=0, fill=1)
        canvas.setFont(self.font_bold, 10)
        canvas.setFillColor(colors.white)
        canvas.drawString(2 * cm, A4[1] - 0.78 * cm, 'RADAR 智能影像诊断平台')
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 0.78 * cm,
                               'AI 辅助诊断报告 · 仅供临床参考')
        # 底部页码
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, 1.45 * cm, A4[0] - 2 * cm, 1.45 * cm)
        canvas.setFont(self.font_name, 8)
        canvas.setFillColor(TEXT_SUB)
        canvas.drawString(2 * cm, 1.05 * cm, '本报告由人工智能辅助诊断系统自动生成')
        canvas.drawRightString(A4[0] - 2 * cm, 1.05 * cm, f'第 {doc.page} 页')
        canvas.restoreState()

    # ------------------------------------------------------------------ #
    # 报告头
    # ------------------------------------------------------------------ #
    def _create_header(self, diagnosis_result: Dict) -> List:
        els = [Spacer(1, 0.35 * cm)]
        els.append(Paragraph('RADAR 智能影像诊断报告', self.styles['title']))
        els.append(Paragraph(
            'RADAR AI-Assisted Radiology Report · 基于视觉-语言模型的腹部CT辅助诊断',
            self.styles['subtitle']))
        els.append(Spacer(1, 0.3 * cm))

        now = datetime.now()
        report_no = f"RADAR-{now.strftime('%Y%m%d-%H%M%S')}"
        patient_id = diagnosis_result.get('patient_id', '—')
        info = Table(
            [[Paragraph(f'<b>报告编号</b>　{report_no}', self.styles['cell']),
              Paragraph(f'<b>患者ID</b>　{patient_id}', self.styles['cell']),
              Paragraph(f'<b>生成时间</b>　{now.strftime("%Y-%m-%d %H:%M")}',
                        self.styles['cell'])]],
            colWidths=[6.4 * cm, 4.6 * cm, 6 * cm],
        )
        info.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), GRAY_BG),
            ('BOX', (0, 0), (-1, -1), 0.8, BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        els.append(info)
        return els

    # ------------------------------------------------------------------ #
    # 患者与检查信息
    # ------------------------------------------------------------------ #
    def _create_patient_section(self, diagnosis_result: Dict, metadata: Dict) -> List:
        els = self._section('一、患者与检查信息')

        label_map = metadata.get('label_map') or {}
        items = []
        for key, fallback_label in _META_FIELDS:
            value = metadata.get(key)
            if value in (None, ''):
                continue
            if key == 'image_shape' and isinstance(value, (list, tuple)):
                value = ' × '.join(str(v) for v in value)
            label = label_map.get(key, fallback_label)
            items.append((label, value))

        if not items:
            items = [('患者ID', diagnosis_result.get('patient_id', '—')),
                     ('检查类型', metadata.get('modality', 'CT'))]

        # 四列布局：标签|值|标签|值
        rows = []
        for i in range(0, len(items), 2):
            pair = items[i:i + 2]
            row = []
            for label, value in pair:
                row.append(Paragraph(label, self.styles['cellLabel']))
                row.append(Paragraph(str(value), self.styles['cell']))
            while len(row) < 4:
                row.append('')
            rows.append(row)

        table = Table(rows, colWidths=[3 * cm, 5.5 * cm, 3 * cm, 5.5 * cm],
                      hAlign='LEFT')
        style = [
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ]
        # 标签列浅灰底
        for r in range(len(rows)):
            style.append(('BACKGROUND', (0, r), (0, r), GRAY_BG))
            style.append(('BACKGROUND', (2, r), (2, r), GRAY_BG))
        table.setStyle(TableStyle(style))
        els.append(table)
        return els

    # ------------------------------------------------------------------ #
    # AI 诊断摘要（指标卡）
    # ------------------------------------------------------------------ #
    def _create_summary_section(self, diagnosis_result: Dict) -> List:
        els = self._section('二、AI 诊断摘要')
        diagnoses = diagnosis_result.get('diagnoses') or []
        explanation = diagnosis_result.get('explanation') or {}
        organs = diagnosis_result.get('organs_detected') or []
        affected = {d.get('organ') for d in diagnoses if d.get('organ')}
        sev = overall_severity(diagnoses)

        proc = explanation.get('processing_time_sec')
        cards = [
            (SEV_LABELS.get(sev, '未见异常'), '总体风险分级',
             SEV_COLORS.get(sev, colors.HexColor('#27ae60'))),
            (str(len(diagnoses)), '阳性发现数', None),
            (str(len(affected)), '受累器官数', None),
            (str(len(organs)), '分割识别器官', None),
            (f'{proc:.1f}s' if isinstance(proc, (int, float)) else '—', 'AI推理耗时', None),
            (str(explanation.get('device') or '—'), '计算设备', None),
        ]

        row_value, row_label = [], []
        for value, label, color in cards:
            vs = ParagraphStyle('v', parent=self.styles['summaryValue'],
                                textColor=color or TEXT_MAIN,
                                fontSize=13 if len(str(value)) > 6 else 15)
            row_value.append(Paragraph(str(value), vs))
            row_label.append(Paragraph(label, self.styles['summaryLabel']))

        col_w = CONTENT_WIDTH / len(cards)
        table = Table([row_value, row_label], colWidths=[col_w] * len(cards))
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), GRAY_BG),
            ('BOX', (0, 0), (-1, -1), 0.8, BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 1),
            ('TOPPADDING', (0, 1), (-1, 1), 1),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 8),
        ]))
        els.append(table)
        return els

    # ------------------------------------------------------------------ #
    # 影像学发现明细表
    # ------------------------------------------------------------------ #
    def _create_diagnosis_section(self, diagnosis_result: Dict) -> List:
        els = self._section('三、影像学发现')
        diagnoses = diagnosis_result.get('diagnoses') or []
        threshold = (diagnosis_result.get('settings') or {}).get('confidence_threshold')

        if threshold is not None:
            els.append(Paragraph(
                f'置信度阈值 {float(threshold):.0%}，以下为达到阈值的阳性发现，'
                f'按置信度降序排列。', self.styles['small']))
            els.append(Spacer(1, 4))

        header = [
            Paragraph('序号', self.styles['cellHead']),
            Paragraph('风险', self.styles['cellHead']),
            Paragraph('器官', self.styles['cellHead']),
            Paragraph('影像学发现', self.styles['cellHead']),
            Paragraph('置信度', self.styles['cellHead']),
            Paragraph('临床建议', self.styles['cellHead']),
        ]
        col_widths = [1.1 * cm, 1.7 * cm, 2.1 * cm, 4.1 * cm, 3.0 * cm, 5.0 * cm]

        if not diagnoses:
            table = Table([header, [
                '', '', '',
                Paragraph('未见达到阈值的阳性影像学发现', self.styles['cell']),
                '', '']], colWidths=col_widths, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
                ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
                ('SPAN', (3, 1), (5, 1)),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            els.append(table)
            return els

        rows = [header]
        row_styles = [
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]
        for idx, d in enumerate(diagnoses):
            sev = d.get('severity') or 'low'
            conf = float(d.get('confidence') or 0)
            # 置信度文字 + 方块进度条（█ 在中文字体中均有字形）
            filled = max(1, int(round(conf * 10)))
            bar = '█' * filled
            finding = d.get('finding', '')
            finding_en = d.get('finding_en') or ''
            finding_html = finding + (
                f'<br/><font size="7.5" color="#5a6b7d">{finding_en}</font>'
                if finding_en else '')
            rows.append([
                Paragraph(str(idx + 1), self.styles['cell']),
                Paragraph(SEV_LABELS.get(sev, '—'), ParagraphStyle(
                    'sev', parent=self.styles['cellBold'],
                    textColor=SEV_COLORS.get(sev, TEXT_MAIN))),
                Paragraph(d.get('organ', ''), self.styles['cellBold']),
                Paragraph(finding_html, self.styles['cell']),
                Paragraph(f'<b>{conf:.1%}</b><br/>'
                          f'<font size="7.5" color="{SEV_HEX.get(sev, "#27ae60")}">{bar}</font>',
                          self.styles['cell']),
                Paragraph(d.get('recommendation', '建议专科进一步评估。'),
                          self.styles['cell']),
            ])
            r = idx + 1
            row_styles.append(('BACKGROUND', (0, r), (-1, r), SEV_ROW_BG.get(sev, colors.white)))

        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle(row_styles))
        # 短表整表不跨页；长表允许分页（表头 repeatRows 自动重复），
        # 否则整表被推到下一页、上一页留下大片空白
        if len(rows) <= 8:
            els.append(KeepTogether(table))
        else:
            els.append(table)
        return els

    # ------------------------------------------------------------------ #
    # 影像证据（CT 截图）
    # ------------------------------------------------------------------ #
    def _create_evidence_section(self, evidence_images: List[Dict]) -> List:
        els = self._section('四、影像证据')
        els.append(Paragraph(
            '以下截图取自本次检查CT体数据（模型输入空间），每组为左右对照：'
            '左图叠加模型原始输出分割（标记受累器官范围），右图为原始影像并以红圈'
            '标出病灶定位，供医师对照上表发现复核。', self.styles['small']))
        els.append(Spacer(1, 6))

        # 每组对照图独占一行（图本身已是左右拼接），图下配说明文字
        for item in evidence_images:
            try:
                from PIL import Image as PILImage
                png = item['png']
                w, h = PILImage.open(io.BytesIO(png)).size
                disp_w = CONTENT_WIDTH
                img = Image(io.BytesIO(png), width=disp_w, height=disp_w * h / w)
                cell = [img, Spacer(1, 3),
                        Paragraph(item.get('title', ''), self.styles['caption'])]
            except Exception:
                continue
            row_table = Table([[cell]], colWidths=[CONTENT_WIDTH])
            row_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ]))
            els.append(row_table)
        return els

    # ------------------------------------------------------------------ #
    # 兼容旧接口：三视图切片 / 统计图表
    # ------------------------------------------------------------------ #
    def _create_image_section(self, image_slices: Dict) -> List:
        els = self._section('影像展示（三视图）')
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        views = [('axial', 'Axial'), ('sagittal', 'Sagittal'), ('coronal', 'Coronal')]
        present = [(k, name) for k, name in views if image_slices.get(k) is not None]
        if not present:
            return els
        fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 4))
        axes = np.atleast_1d(axes)
        for ax, (key, name) in zip(axes, present):
            ax.imshow(np.asarray(image_slices[key]), cmap='gray')
            ax.set_title(name)
            ax.axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        els.append(Image(buf, width=CONTENT_WIDTH,
                         height=CONTENT_WIDTH * 4 / (4 * len(present))))
        return els

    def _create_statistics_section(self, charts: Dict) -> List:
        els = self._section('统计分析')
        for name, fig in charts.items():
            if fig is None:
                continue
            buf = io.BytesIO()
            # matplotlib figure：savefig（旧版误用 plotly 的 write_image 导致崩溃）
            fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
            buf.seek(0)
            img = Image(buf, width=14 * cm, height=9 * cm)
            els.append(KeepTogether([img, Spacer(1, 6)]))
        return els

    # ------------------------------------------------------------------ #
    # 方法学与局限性
    # ------------------------------------------------------------------ #
    def _create_explanation_section(self, diagnosis_result: Dict) -> List:
        explanation = diagnosis_result.get('explanation') or {}
        if not explanation:
            return []
        els = self._section('五、方法学说明与局限性')
        if explanation.get('method'):
            els.append(Paragraph(
                f"<b>分析方法</b>　{explanation['method']}", self.styles['body']))
            els.append(Spacer(1, 3))
        if explanation.get('summary'):
            els.append(Paragraph(
                f"<b>结果解读</b>　{explanation['summary']}", self.styles['body']))
            els.append(Spacer(1, 3))
        limitations = explanation.get('limitations') or []
        if limitations:
            els.append(Paragraph('<b>局限性</b>', self.styles['body']))
            for lim in limitations:
                els.append(Paragraph(f'• {lim}', ParagraphStyle(
                    'lim', parent=self.styles['body'], leftIndent=12, leading=15)))
        return els

    # ------------------------------------------------------------------ #
    # 结论、建议与签署
    # ------------------------------------------------------------------ #
    def _create_conclusion_section(self, diagnosis_result: Dict) -> List:
        els = self._section('六、诊断结论与建议')
        diagnoses = diagnosis_result.get('diagnoses') or []

        if not diagnoses:
            conclusion = ('本次CT检查经AI辅助分析，未检出达到置信度阈值的阳性影像学发现。'
                          '扫描范围内所见器官形态、密度未见明显异常征象。'
                          '建议结合临床病史常规随访；如有症状，请及时专科就诊。')
        else:
            high = [d for d in diagnoses if d.get('severity') == 'high']
            medium = [d for d in diagnoses if d.get('severity') == 'medium']
            top_names = '、'.join(
                f"{d.get('organ', '')}{d.get('finding', '')}（{float(d.get('confidence', 0)):.0%}）"
                for d in diagnoses[:3])
            if high:
                conclusion = (f'AI检出 {len(diagnoses)} 项阳性影像学发现，其中高风险 {len(high)} 项，'
                              f'主要为：{top_names}{"等" if len(diagnoses) > 3 else ""}。'
                              f'建议尽快由专科医师复核原始影像，结合实验室检查与临床病史进一步明确诊断，'
                              f'必要时行增强CT/MRI或穿刺活检。')
            elif medium:
                conclusion = (f'AI检出 {len(diagnoses)} 项阳性影像学发现，以中风险为主，'
                              f'主要为：{top_names}{"等" if len(diagnoses) > 3 else ""}。'
                              f'建议近期专科门诊复查，动态随访观察变化。')
            else:
                conclusion = (f'AI检出 {len(diagnoses)} 项低置信度影像学发现：{top_names}'
                              f'{"等" if len(diagnoses) > 3 else ""}。'
                              f'多为良性或不确定性表现，建议结合临床定期随访。')

        els.append(Paragraph(conclusion, self.styles['body']))
        els.append(Spacer(1, 6))

        # 分项建议（去重保序）
        recs, seen = [], set()
        for d in diagnoses:
            rec = (d.get('recommendation') or '').strip()
            if rec and rec not in seen:
                seen.add(rec)
                recs.append(rec)
        if recs:
            els.append(Paragraph('<b>分项随访建议</b>', self.styles['body']))
            for rec in recs:
                els.append(Paragraph(f'• {rec}', ParagraphStyle(
                    'rec', parent=self.styles['body'], leftIndent=12, leading=15)))
            els.append(Spacer(1, 6))

        # 签署栏（预留医师手写签名位）
        sig = Table([
            [Paragraph('报告生成：RADAR智能影像诊断平台（AI自动生成）', self.styles['cell']),
             Paragraph('审核医师签名：＿＿＿＿＿＿＿＿＿＿', self.styles['cell'])],
            [Paragraph(f'生成时间：{datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}',
                       self.styles['cell']),
             Paragraph('审核日期：＿＿＿＿年＿＿月＿＿日', self.styles['cell'])],
        ], colWidths=[9 * cm, 8 * cm])
        sig.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        els.append(sig)
        return els

    def _create_footer_section(self) -> List:
        disc = Table([[Paragraph(
            '<b>免责声明</b>　本报告由AI辅助诊断系统自动生成，结果反映模型在当前置信度阈值下的输出，'
            '受训练数据分布、扫描质量与个体差异影响，可能存在假阳性或假阴性。'
            '本报告不能替代执业医师的诊断意见，正式诊断结论须由医师复核原始影像并结合临床作出。',
            self.styles['small'])]], colWidths=[CONTENT_WIDTH])
        disc.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff8e6')),
            ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor('#e6c96a')),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        return [disc]


def create_quick_pdf_report(diagnosis_result: Dict,
                            output_path: str,
                            image_slices: Optional[Dict] = None) -> str:
    """快速生成PDF报告（兼容旧调用）。"""
    generator = PDFReportGenerator()
    return generator.generate_report(output_path, diagnosis_result, image_slices)


if __name__ == '__main__':
    # 冒烟测试：中文字体 + 各章节渲染
    test_result = {
        'patient_id': 'TEST001',
        'organs_detected': ['肝', '肾', '脾', '胰腺', '胆囊', '主动脉'],
        'diagnoses': [
            {'organ': '肝', 'finding': '肝囊肿', 'finding_en': 'hepatic cyst',
             'confidence': 0.92, 'severity': 'low',
             'recommendation': '结合既往影像随访；如有增大或症状，建议专科评估。'},
            {'organ': '肾', 'finding': '囊肿', 'confidence': 0.85, 'severity': 'medium',
             'recommendation': '建议结合超声或增强检查由专科医生确认。'},
        ],
        'explanation': {
            'method': '测试方法说明', 'summary': '测试解读',
            'limitations': ['模型输出为器官级诊断项概率，未提供像素级病灶热力图'],
            'processing_time_sec': 12.3, 'device': 'cuda:0',
        },
        'settings': {'confidence_threshold': 0.7},
    }
    gen = PDFReportGenerator()
    out = gen.generate_report('test_report.pdf', test_result,
                              metadata={'patient_id': 'TEST001', 'modality': 'CT',
                                        'image_shape': [230, 512, 512]})
    print(f'字体: {gen.font_name}/{gen.font_bold}, 输出: {out}')
