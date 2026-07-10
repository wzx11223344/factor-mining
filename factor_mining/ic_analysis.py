"""
ic_analysis.py - 因子IC分析模块
=================================

本模块提供基于Spearman秩相关系数的因子IC分析功能，
包括IC均值、ICIR、IC t统计量、IC胜率、IC累计曲线，
以及5分位组合收益分析。

主要功能:
    - Spearman IC计算（逐期）
    - ICIR (IC均值/IC标准差)
    - IC t统计量
    - IC胜率
    - IC累计曲线
    - 5分位组合收益分析

IC (Information Coefficient) 是衡量因子预测能力的核心指标。
Spearman秩相关IC对极端值更稳健，是业界标准。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class ICAnalyzer:
    """因子IC分析器。

    提供Spearman秩IC分析、ICIR计算、IC胜率、IC累计曲线和分位数组合分析。

    Attributes:
        factor_df: 因子面板数据
        forward_days: 未来收益天数
    """

    def __init__(self, factor_df: pd.DataFrame, forward_days: int = 1) -> None:
        """初始化IC分析器。

        Args:
            factor_df: 因子面板数据，需包含 date, symbol, close 列和因子列
            forward_days: 计算IC时使用的未来收益天数
        """
        self.factor_df = factor_df.sort_values(["symbol", "date"]).copy()
        self.forward_days = forward_days
        self._prepare_forward_returns()

    def _prepare_forward_returns(self) -> None:
        """预处理：计算未来收益率。"""
        df = self.factor_df
        df["forward_return"] = (
            df.groupby("symbol")["close"]
            .pct_change(self.forward_days)
            .shift(-self.forward_days)
        )
        self.factor_df = df

    def compute_ic_series(self, factor_name: str) -> pd.Series:
        """计算因子逐期Spearman IC序列。

        IC = Spearman(factor_value, forward_return) on each cross-section

        Args:
            factor_name: 因子名称

        Returns:
            IC序列，索引为日期
        """
        df = self.factor_df[["date", "symbol", factor_name, "forward_return"]].dropna()

        if len(df) == 0:
            logger.warning(f"因子{factor_name}无有效数据")
            return pd.Series(dtype=float)

        ic_series = df.groupby("date").apply(
            lambda x: self._compute_spearman_ic(x[factor_name], x["forward_return"])
        )
        ic_series.name = factor_name
        return ic_series

    @staticmethod
    def _compute_spearman_ic(factor: pd.Series, ret: pd.Series) -> float:
        """计算单期Spearman秩相关IC。

        Args:
            factor: 因子值序列
            ret: 收益率序列

        Returns:
            Spearman IC值，范围[-1, 1]
        """
        mask = (~factor.isna()) & (~ret.isna())
        if mask.sum() < 5:
            return np.nan

        f = factor[mask].values
        r = ret[mask].values

        if np.std(f) == 0 or np.std(r) == 0:
            return 0.0

        ic, _ = scipy_stats.spearmanr(f, r)
        if np.isnan(ic):
            return 0.0
        return ic

    def compute_ic_summary(self, factor_name: str) -> Dict[str, float]:
        """计算因子IC的完整统计指标。

        包括: IC均值、IC标准差、ICIR、IC t值、IC胜率、IC绝对值均值

        Args:
            factor_name: 因子名称

        Returns:
            统计指标字典
        """
        ic_series = self.compute_ic_series(factor_name).dropna()

        if len(ic_series) == 0:
            return {
                "factor": factor_name,
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "icir": np.nan,
                "ic_t": np.nan,
                "ic_winrate": np.nan,
                "ic_abs_mean": np.nan,
                "n_periods": 0,
            }

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()

        # ICIR = IC均值 / IC标准差
        icir = ic_mean / ic_std if ic_std > 0 else 0.0

        # IC t统计量
        n = len(ic_series)
        ic_t = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0

        # IC胜率: IC > 0 的比例
        ic_winrate = (ic_series > 0).sum() / n if n > 0 else 0.0

        # IC绝对值均值
        ic_abs_mean = ic_series.abs().mean()

        summary = {
            "factor": factor_name,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "ic_t": ic_t,
            "ic_winrate": ic_winrate,
            "ic_abs_mean": ic_abs_mean,
            "n_periods": n,
        }

        logger.info(
            f"因子{factor_name} IC分析: mean={ic_mean:.4f}, ICIR={icir:.4f}, "
            f"t={ic_t:.4f}, winrate={ic_winrate:.4f}"
        )
        return summary

    def compute_ic_cumulative(self, factor_name: str) -> pd.Series:
        """计算IC累计曲线。

        Args:
            factor_name: 因子名称

        Returns:
            IC累计值序列
        """
        ic_series = self.compute_ic_series(factor_name).dropna()
        return ic_series.cumsum()

    def compute_quantile_returns(
        self, factor_name: str, n_quantiles: int = 5
    ) -> pd.DataFrame:
        """计算分位数组合收益分析。

        将股票按因子值分为n_quantiles组，计算每组的平均未来收益。
        同时计算多空组合收益（最高组减最低组）。

        Args:
            factor_name: 因子名称
            n_quantiles: 分组数，默认5

        Returns:
            DataFrame，每行为一个日期，列为各分位组的平均收益和多空收益
        """
        df = self.factor_df[["date", "symbol", factor_name, "forward_return"]].dropna()

        if len(df) == 0:
            return pd.DataFrame()

        def _quantile_ret(group: pd.DataFrame) -> pd.Series:
            """计算单期分位组合收益。"""
            if len(group) < n_quantiles:
                return pd.Series({f"Q{i+1}": np.nan for i in range(n_quantiles)})

            try:
                groups = pd.qcut(
                    group[factor_name],
                    q=n_quantiles,
                    labels=False,
                    duplicates="drop",
                )
                if groups.nunique() < 2:
                    return pd.Series({f"Q{i+1}": np.nan for i in range(n_quantiles)})

                result = group.groupby(groups)["forward_return"].mean()
                rets = {}
                for i in range(n_quantiles):
                    if i in result.index:
                        rets[f"Q{i+1}"] = result[i]
                    else:
                        rets[f"Q{i+1}"] = np.nan
                return pd.Series(rets)
            except Exception:
                return pd.Series({f"Q{i+1}": np.nan for i in range(n_quantiles)})

        quantile_df = df.groupby("date").apply(_quantile_ret)
        if len(quantile_df) == 0:
            return pd.DataFrame()

        # 计算多空组合收益
        if f"Q{1}" in quantile_df.columns and f"Q{n_quantiles}" in quantile_df.columns:
            quantile_df["long_short"] = (
                quantile_df[f"Q{n_quantiles}"] - quantile_df[f"Q1"]
            )

        # 计算累计收益
        for col in quantile_df.columns:
            if quantile_df[col].notna().sum() > 0:
                quantile_df[f"{col}_cum"] = (1 + quantile_df[col]).cumprod() - 1

        return quantile_df

    def compute_quantile_summary(
        self, factor_name: str, n_quantiles: int = 5
    ) -> Dict[str, float]:
        """计算分位数组合的汇总统计。

        Args:
            factor_name: 因子名称
            n_quantiles: 分组数

        Returns:
            汇总统计字典
        """
        quantile_df = self.compute_quantile_returns(factor_name, n_quantiles)

        if len(quantile_df) == 0:
            return {}

        summary: Dict[str, float] = {"factor": factor_name}
        for col in [f"Q{i+1}" for i in range(n_quantiles)]:
            if col in quantile_df.columns:
                summary[f"{col}_mean"] = quantile_df[col].mean()

        if "long_short" in quantile_df.columns:
            ls = quantile_df["long_short"].dropna()
            summary["long_short_mean"] = ls.mean()
            summary["long_short_sharpe"] = (
                ls.mean() / ls.std() * np.sqrt(252) if ls.std() > 0 else 0.0
            )
            # 多空胜率
            summary["long_short_winrate"] = (ls > 0).sum() / len(ls) if len(ls) > 0 else 0.0

        return summary

    def analyze_all_factors(
        self, factor_names: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """批量分析所有因子。

        Args:
            factor_names: 因子名称列表

        Returns:
            (ic_summary_df, quantile_summary_df) 两个汇总DataFrame
        """
        ic_summaries = []
        quantile_summaries = []

        for name in factor_names:
            logger.info(f"正在分析因子: {name}")
            ic_summary = self.compute_ic_summary(name)
            ic_summaries.append(ic_summary)

            q_summary = self.compute_quantile_summary(name)
            quantile_summaries.append(q_summary)

        ic_summary_df = pd.DataFrame(ic_summaries)
        quantile_summary_df = pd.DataFrame(quantile_summaries)

        logger.info(f"完成{len(factor_names)}个因子的IC分析")
        return ic_summary_df, quantile_summary_df

    def get_ic_matrix(self, factor_names: List[str]) -> pd.DataFrame:
        """获取所有因子的IC序列矩阵。

        Args:
            factor_names: 因子名称列表

        Returns:
            IC矩阵，每列为一个因子
        """
        ic_data = {}
        for name in factor_names:
            ic_data[name] = self.compute_ic_series(name)

        ic_matrix = pd.DataFrame(ic_data)
        return ic_matrix

    def compute_factor_correlation(self, factor_names: List[str]) -> pd.DataFrame:
        """计算因子间的横截面相关性。

        Args:
            factor_names: 因子名称列表

        Returns:
            因子相关性矩阵
        """
        df = self.factor_df[["date", "symbol"] + factor_names].dropna()
        if len(df) == 0:
            return pd.DataFrame()

        # 取截面均值后的相关性
        corr_data = df[factor_names].corr(method="spearman")
        return corr_data
