"""
可视化工具模块
提供诊断结果的表格、图表等可视化功能
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict


class DiagnosisVisualizer:
    """诊断结果可视化工具"""
    
    SEVERITY_COLORS = {
        'high': '#FF4B4B',      # 红色 - 高危
        'medium': '#FFA500',    # 橙色 - 中危
        'low': '#90EE90',       # 绿色 - 低危
        'normal': '#E8E8E8'     # 灰色 - 正常
    }
    
    ORGAN_CATEGORIES = {
        '消化系统': ['肝', '胆囊', '胰腺', '脾', '胃', '大肠', '小肠', '十二指肠', '食管'],
        '泌尿系统': ['肾', '肾上腺', '膀胱'],
        '循环系统': ['心脏', '主动脉', '门静脉', '肺动脉', '髂动脉', '髂静脉', '下腔静脉'],
        '呼吸系统': ['肺', '气管'],
        '骨骼系统': ['颈椎', '胸椎', '腰椎', '肋骨', '骶骨', '股骨', '肱骨', '锁骨', '肩胛骨', '髋关节'],
        '其他': ['脑', '面部', '竖脊肌', '臀肌', '髂腰肌']
    }
    
    def __init__(self):
        pass
    
    def create_diagnosis_table(self, diagnoses: List[Dict]) -> pd.DataFrame:
        """创建诊断结果表格
        
        Args:
            diagnoses: 诊断结果列表
            
        Returns:
            pd.DataFrame: 诊断结果数据框
        """
        if not diagnoses:
            return pd.DataFrame(columns=['器官', '诊断发现', '置信度', '严重程度'])
        
        data = []
        for diag in diagnoses:
            data.append({
                '器官': diag.get('organ', '未知'),
                '诊断发现': diag.get('finding', '未知'),
                '置信度': f"{diag.get('confidence', 0):.1%}",
                '严重程度': diag.get('severity', 'low')
            })
        
        df = pd.DataFrame(data)
        return df
    
    def create_organ_summary_table(self, organs_detected: List[str], total_organs: int = 36) -> pd.DataFrame:
        """创建器官检测摘要表
        
        Args:
            organs_detected: 检测到的器官列表
            total_organs: 总器官数
            
        Returns:
            pd.DataFrame: 器官检测摘要
        """
        summary = {
            '指标': ['检测到的器官数', '覆盖率', '器官系统数'],
            '数值': [
                len(organs_detected),
                f"{len(organs_detected) / total_organs:.1%}",
                len(self._get_organ_systems(organs_detected))
            ]
        }
        
        return pd.DataFrame(summary)
    
    def _get_organ_systems(self, organs: List[str]) -> List[str]:
        """获取器官所属的系统
        
        Args:
            organs: 器官列表
            
        Returns:
            List[str]: 系统列表
        """
        systems = set()
        for organ in organs:
            for system, system_organs in self.ORGAN_CATEGORIES.items():
                if organ in system_organs:
                    systems.add(system)
                    break
        return list(systems)
    
    def create_severity_pie_chart(self, diagnoses: List[Dict]) -> go.Figure:
        """创建严重程度饼图
        
        Args:
            diagnoses: 诊断结果列表
            
        Returns:
            plotly.graph_objects.Figure: 饼图对象
        """
        if not diagnoses:
            fig = go.Figure()
            fig.add_annotation(
                text="无诊断数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # 统计各严重程度数量
        severity_counts = {}
        for diag in diagnoses:
            severity = diag.get('severity', 'low')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 创建饼图
        labels = list(severity_counts.keys())
        values = list(severity_counts.values())
        colors = [self.SEVERITY_COLORS.get(s, '#E8E8E8') for s in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.3
        )])
        
        fig.update_layout(
            title="诊断发现严重程度分布",
            height=400
        )
        
        return fig
    
    def create_organ_system_bar_chart(self, diagnoses: List[Dict]) -> go.Figure:
        """创建器官系统异常统计柱状图
        
        Args:
            diagnoses: 诊断结果列表
            
        Returns:
            plotly.graph_objects.Figure: 柱状图对象
        """
        if not diagnoses:
            fig = go.Figure()
            fig.add_annotation(
                text="无诊断数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # 统计各器官系统的异常数量
        system_counts = {}
        for diag in diagnoses:
            organ = diag.get('organ', '')
            for system, organs in self.ORGAN_CATEGORIES.items():
                if organ in organs:
                    system_counts[system] = system_counts.get(system, 0) + 1
                    break
        
        if not system_counts:
            fig = go.Figure()
            fig.add_annotation(
                text="无器官系统数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # 创建柱状图
        systems = list(system_counts.keys())
        counts = list(system_counts.values())
        
        fig = go.Figure(data=[go.Bar(
            x=systems,
            y=counts,
            marker_color='steelblue'
        )])
        
        fig.update_layout(
            title="各器官系统异常发现数量",
            xaxis_title="器官系统",
            yaxis_title="异常数量",
            height=400
        )
        
        return fig
    
    def create_confidence_histogram(self, diagnoses: List[Dict]) -> go.Figure:
        """创建置信度分布直方图
        
        Args:
            diagnoses: 诊断结果列表
            
        Returns:
            plotly.graph_objects.Figure: 直方图对象
        """
        if not diagnoses:
            fig = go.Figure()
            fig.add_annotation(
                text="无诊断数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        confidences = [diag.get('confidence', 0) * 100 for diag in diagnoses]
        
        fig = go.Figure(data=[go.Histogram(
            x=confidences,
            nbinsx=10,
            marker_color='lightseagreen'
        )])
        
        fig.update_layout(
            title="诊断置信度分布",
            xaxis_title="置信度 (%)",
            yaxis_title="数量",
            height=400
        )
        
        return fig
    
    def format_diagnosis_report(self, result: Dict) -> str:
        """格式化诊断报告为Markdown文本
        
        Args:
            result: 诊断结果字典
            
        Returns:
            str: Markdown格式的报告
        """
        report = f"# 影像诊断报告\n\n"
        report += f"**患者ID**: {result.get('patient_id', '未知')}\n\n"
        report += "---\n\n"
        
        # 器官检测部分
        report += "## 🔍 器官检测\n\n"
        organs = result.get('organs_detected', [])
        if organs:
            report += f"检测到 **{len(organs)}** 个器官:\n\n"
            # 按系统分组
            for system, system_organs in self.ORGAN_CATEGORIES.items():
                detected_in_system = [o for o in organs if o in system_organs]
                if detected_in_system:
                    report += f"**{system}**: {', '.join(detected_in_system)}\n\n"
        else:
            report += "未检测到器官\n\n"
        
        report += "---\n\n"
        
        # 诊断发现部分
        report += "## 📋 诊断发现\n\n"
        diagnoses = result.get('diagnoses', [])
        
        if not diagnoses:
            report += "✅ **未发现明显异常**\n\n"
        else:
            # 按严重程度分组
            high_severity = [d for d in diagnoses if d.get('severity') == 'high']
            medium_severity = [d for d in diagnoses if d.get('severity') == 'medium']
            low_severity = [d for d in diagnoses if d.get('severity') == 'low']
            
            if high_severity:
                report += "### 🔴 高危发现\n\n"
                for diag in high_severity:
                    report += f"- **{diag.get('organ')}**: {diag.get('finding')} "
                    report += f"(置信度: {diag.get('confidence', 0):.1%})\n"
                report += "\n"
            
            if medium_severity:
                report += "### 🟠 中危发现\n\n"
                for diag in medium_severity:
                    report += f"- **{diag.get('organ')}**: {diag.get('finding')} "
                    report += f"(置信度: {diag.get('confidence', 0):.1%})\n"
                report += "\n"
            
            if low_severity:
                report += "### 🟢 低危发现\n\n"
                for diag in low_severity:
                    report += f"- **{diag.get('organ')}**: {diag.get('finding')} "
                    report += f"(置信度: {diag.get('confidence', 0):.1%})\n"
                report += "\n"
        
        report += "---\n\n"
        report += "*注: 本报告由AI自动生成，仅供参考，不能替代专业医师的诊断意见。*\n"
        
        return report


# 测试代码
if __name__ == "__main__":
    visualizer = DiagnosisVisualizer()
    
    # 测试数据
    test_diagnoses = [
        {'organ': '肝', 'finding': '肝囊肿', 'confidence': 0.92, 'severity': 'low'},
        {'organ': '肾', 'finding': '囊肿', 'confidence': 0.85, 'severity': 'low'},
        {'organ': '肺', 'finding': '结节', 'confidence': 0.78, 'severity': 'medium'}
    ]
    
    df = visualizer.create_diagnosis_table(test_diagnoses)
    print(df)
