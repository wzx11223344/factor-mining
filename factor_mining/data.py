"""
data.py - 基于akshare的A股数据获取模块
========================================

本模块提供A股日线数据(OHLCV)的获取功能，支持指定股票池和日期范围。
在离线环境或akshare不可用时，自动切换到模拟数据模式。

主要功能:
    - 获取个股日线行情数据
    - 获取股票池批量数据
    - 获取市值、估值等基本面数据
    - 模拟数据生成（离线fallback）

数据来源: akshare (https://akshare.akfamily.xyz/)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.warning("akshare未安装，将使用模拟数据模式。请运行 pip install akshare 安装。")


# 默认股票池：沪深300部分成分股
DEFAULT_STOCK_POOL: List[str] = [
    "600519",  # 贵州茅台
    "000858",  # 五粮液
    "601318",  # 中国平安
    "600036",  # 招商银行
    "000333",  # 美的集团
    "601166",  # 兴业银行
    "002594",  # 比亚迪
    "600276",  # 恒瑞医药
    "000651",  # 格力电器
    "601888",  # 中国中免
    "600030",  # 中信证券
    "601012",  # 隆基绿能
    "600887",  # 伊利股份
    "000568",  # 泸州老窖
    "002475",  # 立讯精密
    "600031",  # 三一重工
    "601628",  # 中国人寿
    "600009",  # 上海机场
    "000725",  # 京东方A
    "601398",  # 工商银行
]


class DataFetcher:
    """A股数据获取器，支持akshare在线模式和模拟离线模式。

    Attributes:
        stock_pool: 股票代码列表
        start_date: 数据起始日期 (YYYY-MM-DD)
        end_date: 数据结束日期 (YYYY-MM-DD)
        mock_mode: 是否使用模拟数据模式
    """

    def __init__(
        self,
        stock_pool: Optional[List[str]] = None,
        start_date: str = "2022-01-01",
        end_date: str = "2024-12-31",
        mock_mode: Optional[bool] = None,
    ) -> None:
        """初始化数据获取器。

        Args:
            stock_pool: 股票代码列表，如 ['600519', '000858']。为None时使用默认池。
            start_date: 数据起始日期，格式 YYYY-MM-DD
            end_date: 数据结束日期，格式 YYYY-MM-DD
            mock_mode: 是否强制使用模拟数据。None时自动检测。
        """
        self.stock_pool = stock_pool if stock_pool is not None else DEFAULT_STOCK_POOL.copy()
        self.start_date = start_date
        self.end_date = end_date

        if mock_mode is not None:
            self.mock_mode = mock_mode
        else:
            self.mock_mode = not AKSHARE_AVAILABLE

        if self.mock_mode:
            logger.info("数据获取器运行在模拟数据模式")
        else:
            logger.info("数据获取器运行在akshare在线模式")

    def fetch_single_stock(self, symbol: str) -> pd.DataFrame:
        """获取单只股票的日线数据。

        Args:
            symbol: 股票代码，如 '600519'

        Returns:
            DataFrame，包含列: date, open, high, low, close, volume, amount, turnover
            索引为日期(datetime)
        """
        if self.mock_mode:
            return self._generate_mock_data(symbol)

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=self.start_date.replace("-", ""),
                end_date=self.end_date.replace("-", ""),
                adjust="qfq",
            )
            if df is None or len(df) == 0:
                logger.warning(f"股票{symbol}数据为空，使用模拟数据")
                return self._generate_mock_data(symbol)

            # 统一列名
            col_map = {
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover",
            }
            df = df.rename(columns=col_map)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            # 确保必要列存在
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col not in df.columns:
                    logger.warning(f"股票{symbol}缺少列{col}，使用模拟数据")
                    return self._generate_mock_data(symbol)

            if "turnover" not in df.columns:
                df["turnover"] = 0.0

            logger.info(f"成功获取股票{symbol}数据: {len(df)}条")
            return df

        except Exception as e:
            logger.warning(f"获取股票{symbol}数据失败: {e}，使用模拟数据")
            return self._generate_mock_data(symbol)

    def fetch_stock_pool(self) -> Dict[str, pd.DataFrame]:
        """获取整个股票池的日线数据。

        Returns:
            字典，键为股票代码，值为该股票的DataFrame
        """
        data: Dict[str, pd.DataFrame] = {}
        for symbol in self.stock_pool:
            try:
                df = self.fetch_single_stock(symbol)
                if df is not None and len(df) > 0:
                    data[symbol] = df
            except Exception as e:
                logger.error(f"获取股票{symbol}数据失败: {e}")
        logger.info(f"成功获取{len(data)}/{len(self.stock_pool)}只股票的数据")
        return data

    def fetch_panel_data(self) -> pd.DataFrame:
        """获取股票池的面板数据（长格式）。

        Returns:
            DataFrame，包含列: date, symbol, open, high, low, close, volume, amount, turnover
        """
        stock_data = self.fetch_stock_pool()
        panels = []
        for symbol, df in stock_data.items():
            temp = df.copy()
            temp["symbol"] = symbol
            temp = temp.reset_index()
            panels.append(temp)

        if not panels:
            logger.error("未获取到任何股票数据")
            return pd.DataFrame()

        panel = pd.concat(panels, ignore_index=True)
        panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
        logger.info(f"面板数据: {len(panel)}条记录, {panel['symbol'].nunique()}只股票")
        return panel

    def fetch_market_cap(self, symbol: str) -> Tuple[float, float]:
        """获取股票的总市值和流通市值（单位：元）。

        Args:
            symbol: 股票代码

        Returns:
            (total_market_cap, circulating_market_cap) 总市值和流通市值
        """
        if self.mock_mode:
            return self._generate_mock_market_cap(symbol)

        try:
            df = ak.stock_individual_info_em(symbol=symbol)
            if df is not None and len(df) > 0:
                cap_data = df.set_index("item")["value"].to_dict()
                total_cap = float(cap_data.get("总市值", 0))
                circ_cap = float(cap_data.get("流通市值", 0))
                if total_cap > 0:
                    return total_cap, circ_cap
            return self._generate_mock_market_cap(symbol)
        except Exception as e:
            logger.warning(f"获取股票{symbol}市值失败: {e}，使用模拟数据")
            return self._generate_mock_market_cap(symbol)

    def fetch_valuation_data(self, symbol: str) -> Tuple[float, float]:
        """获取股票的市盈率(PE)和市净率(PB)。

        Args:
            symbol: 股票代码

        Returns:
            (pe, pb) 市盈率和市净率
        """
        if self.mock_mode:
            return self._generate_mock_valuation(symbol)

        try:
            df = ak.stock_a_indicator_lg(symbol=symbol)
            if df is not None and len(df) > 0:
                latest = df.iloc[-1]
                pe = float(latest.get("pe_ttm", 0))
                pb = float(latest.get("pb", 0))
                if pe > 0 and pb > 0:
                    return pe, pb
            return self._generate_mock_valuation(symbol)
        except Exception as e:
            logger.warning(f"获取股票{symbol}估值数据失败: {e}，使用模拟数据")
            return self._generate_mock_valuation(symbol)

    def fetch_market_index(self, index_code: str = "000300") -> pd.DataFrame:
        """获取市场指数日线数据，用于计算特质波动率等。

        Args:
            index_code: 指数代码，默认000300(沪深300)

        Returns:
            DataFrame，包含 date, close 等列
        """
        if self.mock_mode:
            return self._generate_mock_index_data(index_code)

        try:
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            if df is not None and len(df) > 0:
                df = df.rename(columns={"date": "date", "close": "close"})
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                # 截取日期范围
                df = df.loc[self.start_date:self.end_date]
                return df
            return self._generate_mock_index_data(index_code)
        except Exception as e:
            logger.warning(f"获取指数{index_code}数据失败: {e}，使用模拟数据")
            return self._generate_mock_index_data(index_code)

    # ===================== 模拟数据生成方法 =====================

    def _generate_mock_data(self, symbol: str) -> pd.DataFrame:
        """生成模拟的股票日线数据。

        基于几何布朗运动模型生成具有真实特征的股价数据。

        Args:
            symbol: 股票代码（用作随机种子）

        Returns:
            模拟的DataFrame
        """
        rng = np.random.RandomState(hash(symbol) % 2**31)
        dates = pd.bdate_range(start=self.start_date, end=self.end_date)
        n = len(dates)

        # 初始价格基于股票代码生成
        init_price = 10 + (hash(symbol) % 1000) / 10.0
        # 日波动率
        daily_vol = 0.015 + rng.random() * 0.015
        # 日漂移
        daily_drift = (rng.random() - 0.3) * 0.0005

        # 几何布朗运动
        returns = rng.normal(daily_drift, daily_vol, n)
        prices = init_price * np.cumprod(1 + returns)

        # 生成OHLCV
        close = prices
        open_ = close * (1 + rng.normal(0, 0.003, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
        volume = rng.randint(5000000, 50000000, n).astype(float)
        amount = volume * close
        turnover = rng.uniform(0.5, 5.0, n)

        df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "turnover": turnover,
        }, index=dates)
        df.index.name = "date"

        return df

    def _generate_mock_market_cap(self, symbol: str) -> Tuple[float, float]:
        """生成模拟市值数据。

        Args:
            symbol: 股票代码

        Returns:
            (总市值, 流通市值)
        """
        rng = np.random.RandomState(hash(symbol + "cap") % 2**31)
        total_cap = rng.uniform(5e9, 2e12)
        circ_ratio = rng.uniform(0.3, 0.95)
        circ_cap = total_cap * circ_ratio
        return total_cap, circ_cap

    def _generate_mock_valuation(self, symbol: str) -> Tuple[float, float]:
        """生成模拟估值数据。

        Args:
            symbol: 股票代码

        Returns:
            (PE, PB)
        """
        rng = np.random.RandomState(hash(symbol + "val") % 2**31)
        pe = rng.uniform(5, 80)
        pb = rng.uniform(0.5, 10)
        return pe, pb

    def _generate_mock_index_data(self, index_code: str) -> pd.DataFrame:
        """生成模拟指数数据。

        Args:
            index_code: 指数代码

        Returns:
            模拟的指数DataFrame
        """
        rng = np.random.RandomState(hash(index_code) % 2**31)
        dates = pd.bdate_range(start=self.start_date, end=self.end_date)
        n = len(dates)

        returns = rng.normal(0.0002, 0.01, n)
        prices = 3000 * np.cumprod(1 + returns)

        df = pd.DataFrame({"close": prices}, index=dates)
        df.index.name = "date"
        return df

    def get_trading_dates(self) -> List[pd.Timestamp]:
        """获取交易日历。

        Returns:
            交易日列表
        """
        dates = pd.bdate_range(start=self.start_date, end=self.end_date)
        return list(dates)
