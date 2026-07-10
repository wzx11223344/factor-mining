"""
report.py - HTML报告生成模块
==============================

本模块生成包含完整因子分析结果的HTML报告，
包括所有因子IC统计、IC累计曲线、衰减曲线、合成因子对比等图表。

报告内容:
    - 因子概览表格
    - 各因子IC累计曲线
    - 分位数组合收益图
    - IC衰减曲线
    - 合成因子对比
    - 因子相关性热力图

图表使用matplotlib生成base64编码的PNG图片，嵌入HTML中。
"""

import base64
import io
import logging
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


class ReportGenerator:
    """HTML报告生成器。

    将因子分析结果整合为一份完整的HTML报告。

    Attributes:
        output_path: HTML报告输出路径
    """

    def __init__(self, output_path: str = "factor_report.html") -> None:
        """初始化报告生成器。

        Args:
            output_path: HTML报告输出路径
        """
        self.output_path = output_path

    def generate_report(
        self,
        ic_summary: pd.DataFrame,
        quantile_summary: pd.DataFrame,
        ic_matrix: pd.DataFrame,
        decay_summary: pd.DataFrame,
        best_periods: pd.DataFrame,
        synth_comparison: pd.DataFrame,
        factor_corr: pd.DataFrame,
        factor_categories: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """生成完整的HTML报告。

        Args:
            ic_summary: IC汇总统计DataFrame
            quantile_summary: 分位数组合汇总DataFrame
            ic_matrix: IC序列矩阵
            decay_summary: 衰减分析汇总DataFrame
            best_periods: 最佳持有期汇总DataFrame
            synth_comparison: 合成因子对比DataFrame
            factor_corr: 因子相关性矩阵
            factor_categories: 因子分类字典

        Returns:
            HTML报告内容字符串
        """
        logger.info("开始生成HTML报告...")

        # 生成图表
        charts: Dict[str, str] = {}

        # 1. IC柱状图
        charts["ic_bar"] = self._plot_ic_bar(ic_summary)

        # 2. ICIR柱状图
        charts["icir_bar"] = self._plot_icir_bar(ic_summary)

        # 3. IC累计曲线
        charts["ic_cumsum"] = self._plot_ic_cumsum(ic_matrix)

        # 4. 衰减曲线
        charts["decay"] = self._plot_decay_curves(decay_summary)

        # 5. 合成因子对比
        charts["synth_compare"] = self._plot_synth_compare(synth_comparison)

        # 6. 因子相关性热力图
        charts["corr_heatmap"] = self._plot_corr_heatmap(factor_corr)

        # 7. 分位数收益图（取第一个因子示例）
        if len(quantile_summary) > 0:
            charts["quantile"] = self._plot_quantile_returns(quantile_summary)

        # 生成HTML
        html = self._build_html(
            ic_summary=ic_summary,
            quantile_summary=quantile_summary,
            best_periods=best_periods,
            synth_comparison=synth_comparison,
            charts=charts,
            factor_categories=factor_categories,
        )

        # 写入文件
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML报告已生成: {self.output_path}")
        return html

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """将matplotlib图表转为base64编码字符串。"""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{img_str}"

    def _plot_ic_bar(self, ic_summary: pd.DataFrame) -> str:
        """绘制IC均值柱状图。"""
        fig, ax = plt.subplots(figsize=(12, 6))

        factors = ic_summary["factor"].values
        ic_means = ic_summary["ic_mean"].values
        colors = ["#2196F3" if v > 0 else "#F44336" for v in ic_means]

        ax.barh(factors, ic_means, color=colors, edgecolor="white", height=0.7)
        ax.set_xlabel("IC Mean", fontsize=12)
        ax.set_title("各因子IC均值", fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)

        # 添加数值标签
        for i, (name, val) in enumerate(zip(factors, ic_means)):
            if not np.isnan(val):
                ax.text(val + 0.001 * np.sign(val), i, f"{val:.4f}",
                        va="center", fontsize=9)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_icir_bar(self, ic_summary: pd.DataFrame) -> str:
        """绘制ICIR柱状图。"""
        fig, ax = plt.subplots(figsize=(12, 6))

        factors = ic_summary["factor"].values
        icirs = ic_summary["icir"].values
        colors = ["#4CAF50" if abs(v) > 0.5 else "#FF9800" for v in icirs]

        ax.barh(factors, icirs, color=colors, edgecolor="white", height=0.7)
        ax.set_xlabel("ICIR", fontsize=12)
        ax.set_title("各因子ICIR (IC均值/IC标准差)", fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.axvline(x=0.5, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(x=-0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.grid(axis="x", alpha=0.3)

        for i, (name, val) in enumerate(zip(factors, icirs)):
            if not np.isnan(val):
                ax.text(val + 0.01 * np.sign(val), i, f"{val:.3f}",
                        va="center", fontsize=9)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_ic_cumsum(self, ic_matrix: pd.DataFrame) -> str:
        """绘制IC累计曲线。"""
        fig, ax = plt.subplots(figsize=(14, 7))

        for col in ic_matrix.columns:
            cumsum = ic_matrix[col].dropna().cumsum()
            if len(cumsum) > 0:
                ax.plot(cumsum.index, cumsum.values, label=col, linewidth=1.2, alpha=0.8)

        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("IC累计值", fontsize=12)
        ax.set_title("各因子IC累计曲线", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_decay_curves(self, decay_summary: pd.DataFrame) -> str:
        """绘制IC衰减曲线。"""
        fig, ax = plt.subplots(figsize=(12, 7))

        for factor in decay_summary["factor"].unique():
            data = decay_summary[decay_summary["factor"] == factor].sort_values("lag")
            ax.plot(data["lag"], data["ic_mean"], label=factor, linewidth=1.2, alpha=0.8, marker="o", markersize=3)

        ax.set_xlabel("持有期（交易日）", fontsize=12)
        ax.set_ylabel("IC均值", fontsize=12)
        ax.set_title("因子IC衰减曲线", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(range(0, int(decay_summary["lag"].max()) + 1, 2))

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_synth_compare(self, synth_comparison: pd.DataFrame) -> str:
        """绘制合成因子与单因子ICIR对比图。"""
        fig, ax = plt.subplots(figsize=(10, 6))

        df = synth_comparison.sort_values("icir", ascending=True)
        factors = df["factor"].values
        icirs = df["icir"].values

        # 区分单因子和合成因子
        colors = []
        for f in factors:
            if "synth" in str(f):
                colors.append("#E91E63")
            else:
                colors.append("#2196F3")

        ax.barh(factors, icirs, color=colors, edgecolor="white", height=0.6)
        ax.set_xlabel("ICIR", fontsize=12)
        ax.set_title("合成因子 vs 单因子 ICIR对比", fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2196F3", label="单因子"),
            Patch(facecolor="#E91E63", label="合成因子"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_corr_heatmap(self, factor_corr: pd.DataFrame) -> str:
        """绘制因子相关性热力图。"""
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(factor_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(len(factor_corr.columns)))
        ax.set_yticks(range(len(factor_corr.index)))
        ax.set_xticklabels(factor_corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(factor_corr.index, fontsize=8)

        # 添加数值
        for i in range(len(factor_corr.index)):
            for j in range(len(factor_corr.columns)):
                val = factor_corr.iloc[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        fig.colorbar(im, ax=ax, label="Spearman相关系数")
        ax.set_title("因子间相关性矩阵", fontsize=14, fontweight="bold")

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_quantile_returns(self, quantile_summary: pd.DataFrame) -> str:
        """绘制分位数组合收益图。"""
        fig, ax = plt.subplots(figsize=(12, 6))

        factors = quantile_summary["factor"].values
        n_factors = len(factors)
        x = np.arange(n_factors)
        width = 0.15

        # 找出Q1-Q5列
        q_cols = [c for c in quantile_summary.columns if c.startswith("Q") and c.endswith("_mean")]
        q_cols.sort()

        colors_list = ["#F44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]

        for i, col in enumerate(q_cols):
            offset = (i - len(q_cols) / 2 + 0.5) * width
            vals = quantile_summary[col].values
            ax.bar(x + offset, vals, width, label=col.replace("_mean", ""),
                   color=colors_list[i % len(colors_list)], alpha=0.85)

        ax.set_xlabel("因子", fontsize=12)
        ax.set_ylabel("平均收益", fontsize=12)
        ax.set_title("分位数组合平均收益", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(factors, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _df_to_html_table(self, df: pd.DataFrame, max_rows: int = 50) -> str:
        """将DataFrame转为HTML表格。"""
        if len(df) > max_rows:
            df = df.head(max_rows)

        # 格式化数值
        formatted = df.copy()
        for col in formatted.select_dtypes(include=[np.floating]).columns:
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:.4f}" if not pd.isna(x) else "-"
            )

        return formatted.to_html(index=False, classes="data-table", escape=False)

    def _build_html(
        self,
        ic_summary: pd.DataFrame,
        quantile_summary: pd.DataFrame,
        best_periods: pd.DataFrame,
        synth_comparison: pd.DataFrame,
        charts: Dict[str, str],
        factor_categories: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """构建完整HTML报告。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alpha因子挖掘分析报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #1a237e 0%, #0277bd 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header .meta {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            color: #1a237e;
            border-left: 4px solid #0277bd;
            padding-left: 12px;
            margin-bottom: 20px;
            font-size: 20px;
        }}
        .section h3 {{
            color: #333;
            margin: 15px 0 10px;
            font-size: 16px;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
        }}
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin: 15px 0;
        }}
        table.data-table th {{
            background: #1a237e;
            color: white;
            padding: 10px 8px;
            text-align: center;
            font-weight: 600;
        }}
        table.data-table td {{
            padding: 8px;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
        }}
        table.data-table tr:nth-child(even) {{
            background: #f5f5f5;
        }}
        table.data-table tr:hover {{
            background: #e3f2fd;
        }}
        .summary-box {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-card {{
            flex: 1;
            min-width: 200px;
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card .label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #1a237e;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 12px;
        }}
        .tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
            margin: 2px;
        }}
        .tag-green {{ background: #4CAF50; color: white; }}
        .tag-orange {{ background: #FF9800; color: white; }}
        .tag-red {{ background: #F44336; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Alpha因子挖掘分析报告</h1>
            <div class="meta">生成时间: {now} | factor-mining engine v0.1.0</div>
        </div>

        <div class="section">
            <h2>一、报告概览</h2>
            <div class="summary-box">
                <div class="summary-card">
                    <div class="label">因子总数</div>
                    <div class="value">{len(ic_summary)}</div>
                </div>
                <div class="summary-card">
                    <div class="label">有效因子(ICIR>0.5)</div>
                    <div class="value">{len(ic_summary[ic_summary.get('icir', pd.Series(dtype=float)).abs() > 0.5]) if 'icir' in ic_summary.columns else 0}</div>
                </div>
                <div class="summary-card">
                    <div class="label">合成方法数</div>
                    <div class="value">{len(synth_comparison[synth_comparison.get('factor', pd.Series()).astype(str).str.contains('synth', na=False)]) if 'factor' in synth_comparison.columns else 0}</div>
                </div>
                <div class="summary-card">
                    <div class="label">分析期数</div>
                    <div class="value">{int(ic_summary['n_periods'].max()) if 'n_periods' in ic_summary.columns and len(ic_summary) > 0 else 0}</div>
                </div>
            </div>
            <p>本报告基于A股市场数据，对{len(ic_summary)}个Alpha因子进行了全面的IC分析、
            因子合成和衰减分析。以下为详细结果。</p>
        </div>

        <div class="section">
            <h2>二、因子IC统计汇总</h2>
            <p>IC (Information Coefficient) 是衡量因子预测能力的核心指标。
            ICIR = IC均值 / IC标准差，反映因子的稳定性。</p>
            {self._df_to_html_table(ic_summary)}

            <h3>2.1 各因子IC均值</h3>
            <div class="chart-container">
                <img src="{charts.get('ic_bar', '')}" alt="IC均值柱状图">
            </div>

            <h3>2.2 各因子ICIR</h3>
            <div class="chart-container">
                <img src="{charts.get('icir_bar', '')}" alt="ICIR柱状图">
            </div>
        </div>

        <div class="section">
            <h2>三、IC累计曲线</h2>
            <p>IC累计曲线反映因子预测能力的持续性和稳定性。
            累计曲线单调上升表示因子持续有效。</p>
            <div class="chart-container">
                <img src="{charts.get('ic_cumsum', '')}" alt="IC累计曲线">
            </div>
        </div>

        <div class="section">
            <h2>四、分位数组合收益分析</h2>
            <p>将股票按因子值分为5组(Q1-Q5)，Q1为因子值最低组，Q5为最高组。
            如果Q5-Q1(多空组合)收益显著为正，说明因子具有单调预测能力。</p>
            <div class="chart-container">
                <img src="{charts.get('quantile', '')}" alt="分位数组合收益">
            </div>
            {self._df_to_html_table(quantile_summary) if len(quantile_summary) > 0 else '<p>无分位数组合数据</p>'}
        </div>

        <div class="section">
            <h2>五、因子衰减分析</h2>
            <p>因子衰减分析展示因子IC随持有期增加的变化。
            衰减越慢的因子适合更长持有期的策略。</p>
            <div class="chart-container">
                <img src="{charts.get('decay', '')}" alt="IC衰减曲线">
            </div>
            <h3>5.1 最佳持有期</h3>
            {self._df_to_html_table(best_periods) if len(best_periods) > 0 else '<p>无衰减数据</p>'}
        </div>

        <div class="section">
            <h2>六、因子合成结果</h2>
            <p>使用等权、IC加权、PCA和最大化ICIR四种方法合成因子，
            对比合成因子与单因子的IC表现。</p>
            <div class="chart-container">
                <img src="{charts.get('synth_compare', '')}" alt="合成因子对比">
            </div>
            {self._df_to_html_table(synth_comparison) if len(synth_comparison) > 0 else '<p>无合成数据</p>'}
        </div>

        <div class="section">
            <h2>七、因子相关性分析</h2>
            <p>因子间相关性矩阵用于评估因子多样性。
            相关性越低的因子组合，合成效果越好。</p>
            <div class="chart-container">
                <img src="{charts.get('corr_heatmap', '')}" alt="因子相关性热力图">
            </div>
        </div>

        <div class="footer">
            <p>Alpha因子挖掘引擎 (factor-mining) | 由 factor-mining engine 生成</p>
            <p>本报告仅供研究参考，不构成投资建议</p>
        </div>
    </div>
</body>
</html>"""
        return html
