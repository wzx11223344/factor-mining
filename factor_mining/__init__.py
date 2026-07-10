"""
Alpha因子挖掘引擎 (factor-mining)
==================================

一个用于A股市场Alpha因子挖掘、IC分析、因子合成与衰减分析的量化研究框架。

主要模块:
    - data:         基于akshare的数据获取模块
    - factors:      12+ Alpha因子计算模块
    - ic_analysis:  Spearman IC/ICIR分析模块
    - synthesis:    因子合成模块（等权/IC加权/PCA/最大化ICIR）
    - decay:        因子衰减分析模块
    - report:       HTML报告生成模块

作者: factor-mining team
版本: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "factor-mining team"

from .data import DataFetcher
from .factors import FactorCalculator
from .ic_analysis import ICAnalyzer
from .synthesis import FactorSynthesizer
from .decay import DecayAnalyzer
from .report import ReportGenerator

__all__ = [
    "DataFetcher",
    "FactorCalculator",
    "ICAnalyzer",
    "FactorSynthesizer",
    "DecayAnalyzer",
    "ReportGenerator",
]
