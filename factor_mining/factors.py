"""
factors.py - Alpha因子计算模块
================================

本模块实现了12+个常用的Alpha因子，涵盖动量、反转、波动率、流动性、规模和估值六大类。
所有因子均基于真实市场数据(OHLCV)计算，不使用随机数据。

因子列表:
    动量类:   ret_1d, momentum_5d, momentum_20d, momentum_12m_1m
    反转类:   reversal_5d, reversal_20d
    波动类:   volatility_20d, idiosyncratic_vol
    流动性:   amihud_illiq, turnover_rate
    规模类:   log_market_cap, circulating_market_cap
    估值类:   ep (市盈率倒数), bp (市净率倒数)

因子值采用横截面标准化(z-score)处理，便于后续IC分析。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# 因子名称常量
FACTOR_NAMES: List[str] = [
    "ret_1d",               # 1日收益率
    "momentum_5d",          # 5日动量
    "momentum_20d",         # 20日动量
    "momentum_12m_1m",      # 12月减1月动量
    "reversal_5d",          # 5日反转
    "reversal_20d",         # 20日反转
    "volatility_20d",       # 20日波动率
    "idiosyncratic_vol",   # 特质波动率
    "amihud_illiq",        # Amihud非流动性
    "turnover_rate",       # 换手率
    "log_market_cap",      # 市值对数
    "circulating_market_cap",  # 流通市值
    "ep",                  # 市盈率倒数
    "bp",                  # 市净率倒数
]

# 因子分类
FACTOR_CATEGORIES: Dict[str, List[str]] = {
    "动量因子": ["ret_1d", "momentum_5d", "momentum_20d", "momentum_12m_1m"],
    "反转因子": ["reversal_5d", "reversal_20d"],
    "波动因子": ["volatility_20d", "idiosyncratic_vol"],
    "流动性因子": ["amihud_illiq", "turnover_rate"],
    "规模因子": ["log_market_cap", "circulating_market_cap"],
    "估值因子": ["ep", "bp"],
}


class FactorCalculator:
    """Alpha因子计算器，基于OHLCV数据计算各类因子。

    Attributes:
        panel_data: 面板数据DataFrame，包含 date, symbol, open, high, low, close, volume, amount, turnover
        market_index: 市场指数数据（用于特质波动率计算）
        market_caps: 股票市值字典 {symbol: (total_cap, circ_cap)}
        valuations: 股票估值字典 {symbol: (pe, pb)}
    """

    def __init__(
        self,
        panel_data: pd.DataFrame,
        market_index: Optional[pd.DataFrame] = None,
        market_caps: Optional[Dict[str, Tuple[float, float]]] = None,
        valuations: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """初始化因子计算器。

        Args:
            panel_data: 面板数据，需包含 date, symbol, open, high, low, close, volume, amount, turnover
            market_index: 市场指数数据（用于特质波动率），需包含 close 列
            market_caps: 市值字典 {symbol: (总市值, 流通市值)}
            valuations: 估值字典 {symbol: (PE, PB)}
        """
        self.panel_data = panel_data.copy()
        if "date" in self.panel_data.columns:
            self.panel_data["date"] = pd.to_datetime(self.panel_data["date"])
        self.market_index = market_index
        self.market_caps = market_caps or {}
        self.valuations = valuations or {}

        # 预处理：计算日收益率
        self._preprocess()

    def _preprocess(self) -> None:
        """预处理数据，计算日收益率等基础指标。"""
        df = self.panel_data.sort_values(["symbol", "date"]).copy()
        df["ret"] = df.groupby("symbol")["close"].pct_change()
        df["log_ret"] = np.log(df["close"] / df.groupby("symbol")["close"].shift(1))
        self.panel_data = df

    def calculate_all_factors(self) -> pd.DataFrame:
        """计算所有因子，返回横截面因子面板。

        Returns:
            DataFrame，每行为一个(date, symbol)组合，列为各因子值。
            因子值已做横截面z-score标准化。
        """
        df = self.panel_data.copy()

        # 动量因子
        df = self._calc_ret_1d(df)
        df = self._calc_momentum_5d(df)
        df = self._calc_momentum_20d(df)
        df = self._calc_momentum_12m_1m(df)

        # 反转因子
        df = self._calc_reversal_5d(df)
        df = self._calc_reversal_20d(df)

        # 波动因子
        df = self._calc_volatility_20d(df)
        df = self._calc_idiosyncratic_vol(df)

        # 流动性因子
        df = self._calc_amihud_illiq(df)
        df = self._calc_turnover_rate(df)

        # 规模因子
        df = self._calc_log_market_cap(df)
        df = self._calc_circulating_market_cap(df)

        # 估值因子
        df = self._calc_ep(df)
        df = self._calc_bp(df)

        # 横截面标准化
        factor_cols = [c for c in df.columns if c in FACTOR_NAMES]
        df = self._cross_sectional_zscore(df, factor_cols)

        logger.info(f"因子计算完成，共{len(factor_cols)}个因子")
        return df

    def _cross_sectional_zscore(
        self, df: pd.DataFrame, factor_cols: List[str]
    ) -> pd.DataFrame:
        """对因子做横截面z-score标准化。

        Args:
            df: 原始因子面板
            factor_cols: 因子列名列表

        Returns:
            标准化后的DataFrame
        """
        for col in factor_cols:
            if col in df.columns:
                df[col] = df.groupby("date")[col].transform(
                    lambda x: self._winsorize_zscore(x)
                )
        return df

    @staticmethod
    def _winsorize_zscore(x: pd.Series, n_sigma: float = 3.0) -> pd.Series:
        """去极值 + z-score标准化。

        Args:
            x: 原始因子值
            n_sigma: 去极值阈值（标准差倍数）

        Returns:
            标准化后的因子值
        """
        if x.std() == 0 or pd.isna(x.std()):
            return x * 0
        median = x.median()
        mad = np.median(np.abs(x - median))
        if mad == 0 or pd.isna(mad):
            # fallback to mean/std
            mu, sigma = x.mean(), x.std()
            if sigma == 0:
                return x * 0
            z = (x - mu) / sigma
        else:
            # MAD-based robust z-score
            z = 1.4826 * (x - median) / mad
        # 去极值
        z = z.clip(-n_sigma, n_sigma)
        return z

    # ===================== 动量因子 =====================

    @staticmethod
    def _calc_ret_1d(df: pd.DataFrame) -> pd.DataFrame:
        """1日收益率因子。"""
        df["ret_1d"] = df.groupby("symbol")["close"].pct_change(1)
        return df

    @staticmethod
    def _calc_momentum_5d(df: pd.DataFrame) -> pd.DataFrame:
        """5日动量因子：过去5日累计收益率。"""
        df["momentum_5d"] = df.groupby("symbol")["close"].pct_change(5)
        return df

    @staticmethod
    def _calc_momentum_20d(df: pd.DataFrame) -> pd.DataFrame:
        """20日动量因子：过去20日累计收益率。"""
        df["momentum_20d"] = df.groupby("symbol")["close"].pct_change(20)
        return df

    @staticmethod
    def _calc_momentum_12m_1m(df: pd.DataFrame) -> pd.DataFrame:
        """12月减1月动量因子（经典Jegadeesh-Titman动量）。

        计算过去252个交易日减去最近21个交易日的收益率，
        约等于过去12个月收益减去最近1个月收益。
        """
        df = df.sort_values(["symbol", "date"])
        ret_252 = df.groupby("symbol")["close"].pct_change(252)
        ret_21 = df.groupby("symbol")["close"].pct_change(21)
        df["momentum_12m_1m"] = (1 + ret_252) / (1 + ret_21) - 1
        return df

    # ===================== 反转因子 =====================

    @staticmethod
    def _calc_reversal_5d(df: pd.DataFrame) -> pd.DataFrame:
        """5日反转因子：负的过去5日收益率。"""
        df["reversal_5d"] = -df.groupby("symbol")["close"].pct_change(5)
        return df

    @staticmethod
    def _calc_reversal_20d(df: pd.DataFrame) -> pd.DataFrame:
        """20日反转因子：负的过去20日收益率。"""
        df["reversal_20d"] = -df.groupby("symbol")["close"].pct_change(20)
        return df

    # ===================== 波动因子 =====================

    @staticmethod
    def _calc_volatility_20d(df: pd.DataFrame) -> pd.DataFrame:
        """20日波动率因子：过去20日日收益率标准差。"""
        df["volatility_20d"] = (
            df.groupby("symbol")["ret"]
            .rolling(window=20, min_periods=10)
            .std()
            .reset_index(level=0, drop=True)
        )
        return df

    def _calc_idiosyncratic_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """特质波动率因子。

        通过个股收益率对市场收益率回归，取残差的标准差。
        特质波动率 = std(stock_ret - beta * market_ret)

        Args:
            df: 含 ret 列的面板数据

        Returns:
            添加了 idiosyncratic_vol 列的DataFrame
        """
        if self.market_index is None or len(self.market_index) == 0:
            # 无市场指数时，用等权组合作为市场代理
            logger.info("无市场指数数据，使用等权组合作为市场代理")
            market_ret = df.groupby("date")["ret"].mean()
            market_ret = market_ret.rename("market_ret")
            df = df.merge(market_ret, on="date", how="left")
        else:
            market_data = self.market_index.copy()
            market_ret = market_data["close"].pct_change()
            market_ret.name = "market_ret"
            market_ret.index.name = "date"
            df = df.merge(
                market_ret.reset_index(), on="date", how="left"
            )

        # 计算特质波动率
        df = df.sort_values(["symbol", "date"])

        def _idio_vol(group: pd.DataFrame) -> pd.Series:
            """计算单只股票的特质波动率。"""
            valid = group.dropna(subset=["ret", "market_ret"])
            if len(valid) < 20:
                return pd.Series(np.nan, index=group.index)

            # 滚动回归
            window = 60
            rets = group["ret"].values
            mrets = group["market_ret"].values
            result = np.full(len(group), np.nan)

            for i in range(window, len(group)):
                r = rets[i - window:i]
                mr = mrets[i - window:i]
                mask = ~np.isnan(r) & ~np.isnan(mr)
                if mask.sum() < 20:
                    continue
                # OLS回归
                slope, intercept = np.polyfit(mr[mask], r[mask], 1)
                residual = r[mask] - (slope * mr[mask] + intercept)
                result[i] = np.std(residual)

            return pd.Series(result, index=group.index)

        df["idiosyncratic_vol"] = df.groupby("symbol").apply(_idio_vol).reset_index(level=0, drop=True)
        return df

    # ===================== 流动性因子 =====================

    @staticmethod
    def _calc_amihud_illiq(df: pd.DataFrame) -> pd.DataFrame:
        """Amihud非流动性因子。

        illiq = mean(|ret| / amount)

        Args:
            df: 面板数据

        Returns:
            添加了 amihud_illiq 列的DataFrame
        """
        df = df.copy()
        df["abs_ret"] = df["ret"].abs()
        # 避免除以0
        df["illiq_daily"] = df["abs_ret"] / (df["amount"].replace(0, np.nan))

        df["amihud_illiq"] = (
            df.groupby("symbol")["illiq_daily"]
            .rolling(window=20, min_periods=10)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df.drop(columns=["abs_ret", "illiq_daily"], inplace=True)
        return df

    @staticmethod
    def _calc_turnover_rate(df: pd.DataFrame) -> pd.DataFrame:
        """换手率因子：过去20日平均换手率。"""
        if "turnover" in df.columns:
            df["turnover_rate"] = (
                df.groupby("symbol")["turnover"]
                .rolling(window=20, min_periods=10)
                .mean()
                .reset_index(level=0, drop=True)
            )
        else:
            # 如果没有换手率数据，用成交量/20日均量代替
            df["turnover_rate"] = df.groupby("symbol")["volume"].pct_change()
        return df

    # ===================== 规模因子 =====================

    def _calc_log_market_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """市值对数因子。"""
        caps = []
        for sym in df["symbol"].unique():
            if sym in self.market_caps:
                total_cap, _ = self.market_caps[sym]
                caps.append({"symbol": sym, "log_market_cap": np.log(total_cap) if total_cap > 0 else np.nan})
            else:
                caps.append({"symbol": sym, "log_market_cap": np.nan})

        cap_df = pd.DataFrame(caps)
        df = df.merge(cap_df, on="symbol", how="left")
        return df

    def _calc_circulating_market_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """流通市值因子（对数）。"""
        caps = []
        for sym in df["symbol"].unique():
            if sym in self.market_caps:
                _, circ_cap = self.market_caps[sym]
                caps.append({"symbol": sym, "circulating_market_cap": np.log(circ_cap) if circ_cap > 0 else np.nan})
            else:
                caps.append({"symbol": sym, "circulating_market_cap": np.nan})

        cap_df = pd.DataFrame(caps)
        df = df.merge(cap_df, on="symbol", how="left")
        return df

    # ===================== 估值因子 =====================

    def _calc_ep(self, df: pd.DataFrame) -> pd.DataFrame:
        """EP因子：市盈率倒数 (1/PE)。"""
        eps = []
        for sym in df["symbol"].unique():
            if sym in self.valuations:
                pe, _ = self.valuations[sym]
                eps.append({"symbol": sym, "ep": 1.0 / pe if pe > 0 else np.nan})
            else:
                eps.append({"symbol": sym, "ep": np.nan})

        ep_df = pd.DataFrame(eps)
        df = df.merge(ep_df, on="symbol", how="left")
        return df

    def _calc_bp(self, df: pd.DataFrame) -> pd.DataFrame:
        """BP因子：市净率倒数 (1/PB)。"""
        bps = []
        for sym in df["symbol"].unique():
            if sym in self.valuations:
                _, pb = self.valuations[sym]
                bps.append({"symbol": sym, "bp": 1.0 / pb if pb > 0 else np.nan})
            else:
                bps.append({"symbol": sym, "bp": np.nan})

        bp_df = pd.DataFrame(bps)
        df = df.merge(bp_df, on="symbol", how="left")
        return df

    def get_factor_returns(
        self, factor_df: pd.DataFrame, factor_name: str, forward_days: int = 1
    ) -> pd.DataFrame:
        """计算因子值与未来收益率的对应关系。

        Args:
            factor_df: 因子面板数据
            factor_name: 因子名称
            forward_days: 未来收益率的天数

        Returns:
            DataFrame，包含 date, symbol, factor_value, forward_return
        """
        df = factor_df.sort_values(["symbol", "date"]).copy()
        df["forward_return"] = df.groupby("symbol")["close"].pct_change(forward_days).shift(-forward_days)

        result = df[["date", "symbol", factor_name, "forward_return"]].dropna()
        return result
