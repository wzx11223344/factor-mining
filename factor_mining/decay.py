"""
decay.py - 因子衰减分析模块
==============================

本模块分析因子预测能力随持有期变化的衰减特征。
通过计算因子在不同滞后期(1-20日)的IC，绘制IC衰减曲线，
并寻找因子的最佳持有期。

主要功能:
    - 计算1-20日不同滞后期下的IC
    - 绘制IC衰减曲线
    - 寻找最佳持有期（ICIR最大的滞后期）
    - 衰减率分析

因子衰减分析是因子投资中的重要环节，帮助确定因子的有效持有期。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class DecayAnalyzer:
    """因子衰减分析器。

    计算因子在不同持有期下的IC变化，评估因子的衰减特征。

    Attributes:
        factor_df: 因子面板数据
        max_lag: 最大滞后期（交易日）
    """

    def __init__(
        self,
        factor_df: pd.DataFrame,
        max_lag: int = 20,
    ) -> None:
        """初始化衰减分析器。

        Args:
            factor_df: 因子面板数据，需包含 date, symbol, close 和因子列
            max_lag: 最大滞后期天数，默认20
        """
        self.factor_df = factor_df.sort_values(["symbol", "date"]).copy()
        self.max_lag = max_lag
        self._prepare_forward_returns()

    def _prepare_forward_returns(self) -> None:
        """预处理：计算各滞后期下的未来收益率。"""
        df = self.factor_df
        for lag in range(1, self.max_lag + 1):
            col_name = f"forward_ret_{lag}d"
            df[col_name] = (
                df.groupby("symbol")["close"]
                .pct_change(lag)
                .shift(-lag)
            )
        self.factor_df = df

    def compute_ic_at_lag(
        self, factor_name: str, lag: int
    ) -> float:
        """计算因子在指定滞后期下的IC。

        Args:
            factor_name: 因子名称
            lag: 滞后期天数

        Returns:
            该滞后期下的IC均值
        """
        ret_col = f"forward_ret_{lag}d"
        if ret_col not in self.factor_df.columns:
            logger.warning(f"未找到{ret_col}列")
            return np.nan

        df = self.factor_df[["date", "symbol", factor_name, ret_col]].dropna()
        if len(df) == 0:
            return np.nan

        ic_series = df.groupby("date").apply(
            lambda x: self._spearman_ic(x[factor_name], x[ret_col])
        ).dropna()

        if len(ic_series) == 0:
            return np.nan

        return ic_series.mean()

    @staticmethod
    def _spearman_ic(factor: pd.Series, ret: pd.Series) -> float:
        """计算Spearman秩相关IC。"""
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

    def compute_ic_decay(
        self, factor_name: str
    ) -> pd.DataFrame:
        """计算因子的IC衰减曲线。

        计算因子在1到max_lag各滞后期下的IC均值、IC标准差和ICIR。

        Args:
            factor_name: 因子名称

        Returns:
            DataFrame，列为 lag, ic_mean, ic_std, icir
        """
        results: List[Dict] = []

        for lag in range(1, self.max_lag + 1):
            ret_col = f"forward_ret_{lag}d"
            df = self.factor_df[["date", "symbol", factor_name, ret_col]].dropna()

            if len(df) == 0:
                results.append({
                    "lag": lag,
                    "ic_mean": np.nan,
                    "ic_std": np.nan,
                    "icir": np.nan,
                    "n_periods": 0,
                })
                continue

            ic_series = df.groupby("date").apply(
                lambda x: self._spearman_ic(x[factor_name], x[ret_col])
            ).dropna()

            if len(ic_series) == 0:
                results.append({
                    "lag": lag,
                    "ic_mean": np.nan,
                    "ic_std": np.nan,
                    "icir": np.nan,
                    "n_periods": 0,
                })
                continue

            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0.0

            results.append({
                "lag": lag,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "n_periods": len(ic_series),
            })

        decay_df = pd.DataFrame(results)
        logger.info(
            f"因子{factor_name}衰减分析完成: "
            f"IC从{decay_df['ic_mean'].iloc[0]:.4f}(1d)到{decay_df['ic_mean'].iloc[-1]:.4f}({self.max_lag}d)"
        )
        return decay_df

    def find_optimal_holding_period(
        self, factor_name: str
    ) -> Tuple[int, float]:
        """找到因子的最佳持有期。

        最佳持有期定义为ICIR最大的滞后期。

        Args:
            factor_name: 因子名称

        Returns:
            (最佳持有期天数, 该持有期的ICIR)
        """
        decay_df = self.compute_ic_decay(factor_name)
        valid = decay_df.dropna(subset=["icir"])

        if len(valid) == 0:
            logger.warning(f"因子{factor_name}无法找到最佳持有期")
            return 1, 0.0

        best_idx = valid["icir"].abs().idxmax()
        best_lag = int(valid.loc[best_idx, "lag"])
        best_icir = float(valid.loc[best_idx, "icir"])

        logger.info(
            f"因子{factor_name}最佳持有期: {best_lag}天, ICIR={best_icir:.4f}"
        )
        return best_lag, best_icir

    def compute_decay_rate(
        self, factor_name: str
    ) -> float:
        """计算因子的IC衰减率。

        衰减率 = (IC_1d - IC_max_lag) / IC_1d

        Args:
            factor_name: 因子名称

        Returns:
            IC衰减率，正值表示衰减
        """
        decay_df = self.compute_ic_decay(factor_name)

        ic_1d = decay_df.iloc[0]["ic_mean"]
        ic_max = decay_df.iloc[-1]["ic_mean"]

        if pd.isna(ic_1d) or ic_1d == 0:
            return np.nan

        decay_rate = (ic_1d - ic_max) / abs(ic_1d)
        logger.info(f"因子{factor_name} IC衰减率: {decay_rate:.4f}")
        return decay_rate

    def analyze_all_factors(
        self, factor_names: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """批量分析所有因子的衰减特征。

        Args:
            factor_names: 因子名称列表

        Returns:
            (衰减曲线汇总DataFrame, 最佳持有期汇总DataFrame)
        """
        all_decay: List[pd.DataFrame] = []
        best_periods: List[Dict] = []

        for name in factor_names:
            logger.info(f"正在分析因子衰减: {name}")
            decay_df = self.compute_ic_decay(name)
            decay_df["factor"] = name
            all_decay.append(decay_df)

            best_lag, best_icir = self.find_optimal_holding_period(name)
            best_periods.append({
                "factor": name,
                "best_holding_period": best_lag,
                "best_icir": best_icir,
                "ic_1d": decay_df.iloc[0]["ic_mean"],
                "ic_decay_rate": self.compute_decay_rate(name),
            })

        decay_summary = pd.concat(all_decay, ignore_index=True)
        best_period_df = pd.DataFrame(best_periods)

        logger.info(f"完成{len(factor_names)}个因子的衰减分析")
        return decay_summary, best_period_df

    def get_half_life(
        self, factor_name: str
    ) -> float:
        """计算因子的IC半衰期。

        半衰期定义为IC衰减到初始IC一半时对应的持有期。

        Args:
            factor_name: 因子名称

        Returns:
            半衰期天数，如果IC始终未衰减到一半则返回max_lag
        """
        decay_df = self.compute_ic_decay(factor_name)
        ic_1d = decay_df.iloc[0]["ic_mean"]

        if pd.isna(ic_1d) or ic_1d == 0:
            return np.nan

        half_ic = ic_1d / 2.0

        for _, row in decay_df.iterrows():
            if pd.isna(row["ic_mean"]):
                continue
            if abs(row["ic_mean"]) <= abs(half_ic):
                half_life = row["lag"]
                logger.info(f"因子{factor_name} IC半衰期: {half_life}天")
                return float(half_life)

        logger.info(f"因子{factor_name} IC在{self.max_lag}日内未衰减到一半")
        return float(self.max_lag)
