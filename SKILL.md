# SKILL.md - Alpha因子挖掘引擎技能说明

## 概述

本技能 (skill) 提供一套完整的A股Alpha因子挖掘、分析和报告生成流程。通过CLI工具 `mine.py` 一键执行从数据获取到报告生成的完整pipeline。

## 技能能力

### 核心能力
1. **数据获取**: 通过akshare获取A股日线行情、市值、估值数据
2. **因子计算**: 计算14个涵盖动量、反转、波动、流动性、规模、估值的Alpha因子
3. **IC分析**: Spearman IC、ICIR、t统计量、胜率、分位数组合收益
4. **因子合成**: 等权、IC加权、PCA、最大化ICIR四种合成方法
5. **衰减分析**: 1-20日IC衰减曲线、最佳持有期、半衰期
6. **报告生成**: 包含全部图表的HTML报告

### 因子清单 (14个)

#### 动量因子 (4个)
- `ret_1d`: 1日收益率
- `momentum_5d`: 5日累计收益率
- `momentum_20d`: 20日累计收益率
- `momentum_12m_1m`: 12月减1月动量 (Jegadeesh-Titman)

#### 反转因子 (2个)
- `reversal_5d`: 负5日收益率
- `reversal_20d`: 负20日收益率

#### 波动因子 (2个)
- `volatility_20d`: 20日日收益率标准差
- `idiosyncratic_vol`: 个股收益率对市场回归的残差标准差

#### 流动性因子 (2个)
- `amihud_illiq`: |收益率| / 成交额 的20日均值
- `turnover_rate`: 20日平均换手率

#### 规模因子 (2个)
- `log_market_cap`: 总市值的自然对数
- `circulating_market_cap`: 流通市值的自然对数

#### 估值因子 (2个)
- `ep`: 1/PE (市盈率倒数)
- `bp`: 1/PB (市净率倒数)

## 使用方法

### 基本用法

```bash
# 模拟数据模式（离线）
python mine.py --mock

# 在线模式
python mine.py --start 2022-01-01 --end 2024-12-31
```

### 高级选项

```bash
# 指定股票池
python mine.py --stocks 600519 000858 601318 600036

# 指定输出和衰减分析参数
python mine.py --output report.html --max-decay-lag 30

# 详细日志
python mine.py --mock --verbose
```

### Python API调用

```python
from factor_mining import DataFetcher, FactorCalculator, ICAnalyzer

# 1. 获取数据
fetcher = DataFetcher(start_date="2022-01-01", end_date="2024-12-31", mock_mode=True)
panel = fetcher.fetch_panel_data()
index = fetcher.fetch_market_index()

# 2. 计算因子
calc = FactorCalculator(panel, market_index=index)
factor_df = calc.calculate_all_factors()

# 3. IC分析
analyzer = ICAnalyzer(factor_df)
summary = analyzer.compute_ic_summary("momentum_20d")
```

## 输出

- **HTML报告**: 包含IC统计表、IC柱状图、ICIR图、IC累计曲线、衰减曲线、合成因子对比、相关性热力图
- **日志输出**: 每个步骤的执行状态和关键指标

## 依赖

- numpy, pandas, scipy, scikit-learn (核心)
- akshare (数据获取，离线模式可选)
- matplotlib (可视化)

## 模拟模式

当akshare不可用或需要离线运行时，使用 `--mock` 参数启动模拟模式。
模拟数据基于几何布朗运动模型生成，具有真实的股价特征（波动率、漂移率等），
确保因子计算和IC分析的流程完整性。

## 注意事项

1. 在线模式下akshare可能因网络或接口变更导致部分数据获取失败，系统会自动回退到模拟数据
2. 因子值均经过横截面MAD去极值 + z-score标准化处理
3. IC分析使用Spearman秩相关系数，对极端值更稳健
4. 特质波动率需要市场指数数据，无指数时使用等权组合作为市场代理
5. 本工具仅供研究使用，不构成投资建议
