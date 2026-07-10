"""
synthesis.py - 因子合成模块
=============================

本模块提供多种因子合成方法，将多个单因子合成为综合因子，
以提升预测能力和降低单因子噪声。

合成方法:
    - 等权合成 (Equal Weight)
    - IC加权合成 (IC Weighted)
    - PCA合成 (第一主成分)
    - 最大化IC_IR合成 (简化版)

合成后会对合成因子进行IC评估，比较合成效果。

合成因子是因子投资中的关键环节，好的合成方法可以显著提升因子收益。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .ic_analysis import ICAnalyzer

logger = logging.getLogger(__name__)


class FactorSynthesizer:
    """因子合成器，支持多种合成方法。

    Attributes:
        factor_df: 因子面板数据
        factor_names: 参与合成的因子名称列表
        ic_analyzer: IC分析器实例
    """

    def __init__(
        self,
        factor_df: pd.DataFrame,
        factor_names: List[str],
        forward_days: int = 1,
    ) -> None:
        """初始化因子合成器。

        Args:
            factor_df: 因子面板数据，需包含 date, symbol, close 和因子列
            factor_names: 参与合成的因子名称列表
            forward_days: 未来收益天数（用于IC评估）
        """
        self.factor_df = factor_df.copy()
        self.factor_names = factor_names
        self.forward_days = forward_days
        self.ic_analyzer = ICAnalyzer(factor_df, forward_days)

    def equal_weight(self) -> pd.Series:
        """等权合成：对所有因子取简单平均。

        Returns:
            合成因子值序列，索引与factor_df对齐
        """
        available = [f for f in self.factor_names if f in self.factor_df.columns]
        if not available:
            logger.warning("无可用因子进行等权合成")
            return pd.Series(dtype=float)

        # 横截面标准化后等权平均
        combined = self.factor_df[available].mean(axis=1)
        combined.name = "synth_equal_weight"
        logger.info(f"等权合成完成，使用{len(available)}个因子")
        return combined

    def ic_weighted(self) -> pd.Series:
        """IC加权合成：按各因子IC均值占比分配权重。

        权重 = IC_i / sum(IC_j) for all j

        Returns:
            IC加权合成因子值序列
        """
        ic_weights: Dict[str, float] = {}
        total_ic = 0.0

        for name in self.factor_names:
            if name not in self.factor_df.columns:
                continue
            summary = self.ic_analyzer.compute_ic_summary(name)
            ic_mean = summary.get("ic_mean", 0.0)
            # 确保IC方向一致（取绝对值方向，保留符号）
            if not np.isnan(ic_mean):
                ic_weights[name] = ic_mean
                total_ic += abs(ic_mean)

        if total_ic == 0 or len(ic_weights) == 0:
            logger.warning("IC加权失败（IC均为0），回退到等权")
            return self.equal_weight()

        # 归一化权重
        weights = {k: v / total_ic for k, v in ic_weights.items()}

        logger.info("IC加权权重:")
        for k, v in weights.items():
            logger.info(f"  {k}: {v:.4f}")

        # 加权合成
        combined = pd.Series(0.0, index=self.factor_df.index)
        for name, weight in weights.items():
            vals = self.factor_df[name].fillna(0)
            combined += weight * vals

        combined.name = "synth_ic_weighted"
        return combined

    def pca_synthesis(self, n_components: int = 1) -> Tuple[pd.Series, Dict]:
        """PCA合成：提取第一主成分作为合成因子。

        Args:
            n_components: PCA主成分数量（默认取第一主成分）

        Returns:
            (合成因子值, PCA信息字典) 元组
        """
        available = [f for f in self.factor_names if f in self.factor_df.columns]
        if len(available) < 2:
            logger.warning("PCA合成需要至少2个因子")
            return pd.Series(dtype=float), {}

        # 准备数据
        factor_matrix = self.factor_df[available].copy()
        # 按日期分组标准化
        for date in factor_matrix.index.get_level_values("date").unique() if "date" in self.factor_df.columns else [None]:
            if date is not None:
                mask = self.factor_df["date"] == date
                factor_matrix.loc[mask] = StandardScaler().fit_transform(
                    factor_matrix.loc[mask].fillna(0)
                )

        factor_matrix = factor_matrix.fillna(0)

        # PCA拟合
        pca = PCA(n_components=min(n_components, len(available)))
        pca.fit(factor_matrix)

        # 获取第一主成分
        components = pca.components_[0]
        explained_variance = pca.explained_variance_ratio_[0]

        logger.info(
            f"PCA合成完成，第一主成分解释方差比: {explained_variance:.4f}"
        )
        logger.info("PCA载荷:")
        for name, load in zip(available, components):
            logger.info(f"  {name}: {load:.4f}")

        # 投影到第一主成分
        combined = factor_matrix @ components
        combined.name = "synth_pca"

        info = {
            "explained_variance": explained_variance,
            "loadings": dict(zip(available, components)),
            "n_components": len(available),
        }

        return combined, info

    def max_ic_ir(self) -> pd.Series:
        """最大化IC_IR合成（简化版）。

        通过优化因子权重使得合成因子的ICIR最大化。
        简化版使用历史IC序列的均值和协方差矩阵进行近似优化:

            w = Sigma^{-1} * mu / (mu' * Sigma^{-1} * mu)

        其中 mu 是IC均值向量, Sigma 是IC协方差矩阵。

        Returns:
            最大化ICIR合成因子值序列
        """
        available = [f for f in self.factor_names if f in self.factor_df.columns]
        if len(available) < 2:
            logger.warning("最大化ICIR合成需要至少2个因子")
            return self.equal_weight()

        # 获取IC序列矩阵
        ic_matrix = self.ic_analyzer.get_ic_matrix(available).dropna()

        if len(ic_matrix) < 20:
            logger.warning("IC样本不足，回退到IC加权")
            return self.ic_weighted()

        mu = ic_matrix.mean().values  # IC均值向量
        sigma = ic_matrix.cov().values  # IC协方差矩阵

        # 添加正则化避免奇异矩阵
        n = len(available)
        sigma_reg = sigma + np.eye(n) * 1e-6

        try:
            # 最优权重: w = Sigma^{-1} * mu
            w = np.linalg.solve(sigma_reg, mu)

            # 确保权重方向与IC方向一致
            # 如果合成因子IC为负，翻转方向
            combined_ic = np.dot(mu, w)
            if combined_ic < 0:
                w = -w

            # 归一化权重
            w = w / (np.abs(w).sum() if np.abs(w).sum() > 0 else 1.0)

        except np.linalg.LinAlgError:
            logger.warning("矩阵求解失败，回退到IC加权")
            return self.ic_weighted()

        logger.info("最大化ICIR合成权重:")
        for name, weight in zip(available, w):
            logger.info(f"  {name}: {weight:.4f}")

        # 加权合成
        combined = pd.Series(0.0, index=self.factor_df.index)
        for name, weight in zip(available, w):
            vals = self.factor_df[name].fillna(0)
            combined += weight * vals

        combined.name = "synth_max_icir"
        return combined

    def synthesize_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """执行所有合成方法并评估。

        Returns:
            (合成因子DataFrame, 合成因子IC评估DataFrame)
        """
        results: Dict[str, pd.Series] = {
            "equal_weight": self.equal_weight(),
            "ic_weighted": self.ic_weighted(),
            "pca": self.pca_synthesis()[0],
            "max_icir": self.max_ic_ir(),
        }

        # 构建合成因子面板
        synth_df = self.factor_df[["date", "symbol", "close"]].copy()
        for name, series in results.items():
            if len(series) > 0:
                synth_df[f"synth_{name}"] = series.values

        # 评估合成因子IC
        synth_factor_names = [f"synth_{name}" for name in results.keys()]
        synth_ic_summaries = []

        for name in synth_factor_names:
            if name in synth_df.columns:
                summary = self.ic_analyzer.compute_ic_summary(name)
                synth_ic_summaries.append(summary)

        ic_eval_df = pd.DataFrame(synth_ic_summaries)

        logger.info("因子合成完成，共4种方法")
        return synth_df, ic_eval_df

    def compare_with_single_factors(
        self, single_ic_df: pd.DataFrame
    ) -> pd.DataFrame:
        """将合成因子IC与单因子IC对比。

        Args:
            single_ic_df: 单因子IC汇总DataFrame

        Returns:
            对比DataFrame
        """
        _, synth_ic_df = self.synthesize_all()

        comparison = pd.concat([single_ic_df, synth_ic_df], ignore_index=True)
        comparison = comparison.sort_values("icir", ascending=False)

        logger.info("合成因子与单因子IC对比完成")
        return comparison
