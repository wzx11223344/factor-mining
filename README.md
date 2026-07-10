# Alpha因子挖掘引擎 (factor-mining)

一个用于A股市场Alpha因子挖掘、IC分析、因子合成与衰减分析的Python量化研究框架。

## 项目结构

```
factor-mining/
├── mine.py              # CLI入口
├── factor_mining/
│   ├── __init__.py
│   ├── data.py          # akshare数据获取
│   ├── factors.py       # 14个Alpha因子计算
│   ├── ic_analysis.py   # Spearman IC/ICIR分析
│   ├── synthesis.py     # 因子合成(等权/IC加权/PCA/最大化ICIR)
│   ├── decay.py         # 因子衰减分析
│   └── report.py        # HTML报告生成
├── README.md
├── SKILL.md
└── requirements.txt
```

## 功能特性

### 1. 数据获取 (data.py)
- 基于akshare获取A股日线OHLCV数据
- 支持自定义股票池和日期范围
- 自动获取市值、估值(PE/PB)数据
- 离线模拟数据模式（基于几何布朗运动）

### 2. Alpha因子 (factors.py) - 14个因子

| 类别 | 因子 | 说明 |
|------|------|------|
| 动量 | ret_1d | 1日收益率 |
| 动量 | momentum_5d | 5日动量 |
| 动量 | momentum_20d | 20日动量 |
| 动量 | momentum_12m_1m | 12月减1月动量 |
| 反转 | reversal_5d | 5日反转 |
| 反转 | reversal_20d | 20日反转 |
| 波动 | volatility_20d | 20日波动率 |
| 波动 | idiosyncratic_vol | 特质波动率 |
| 流动性 | amihud_illiq | Amihud非流动性 |
| 流动性 | turnover_rate | 换手率 |
| 规模 | log_market_cap | 市值对数 |
| 规模 | circulating_market_cap | 流通市值 |
| 估值 | ep | 市盈率倒数 |
| 估值 | bp | 市净率倒数 |

### 3. IC分析 (ic_analysis.py)
- Spearman秩相关系数IC
- ICIR (IC均值/IC标准差)
- IC t统计量
- IC胜率
- IC累计曲线
- 5分位组合收益分析

### 4. 因子合成 (synthesis.py)
- 等权合成
- IC加权合成
- PCA合成（第一主成分）
- 最大化IC_IR合成（简化版）

### 5. 因子衰减分析 (decay.py)
- 1-20日不同滞后期IC计算
- IC衰减曲线绘制
- 最佳持有期识别
- IC半衰期计算

### 6. 报告生成 (report.py)
- 完整HTML报告
- IC柱状图、ICIR图、累计曲线
- 衰减曲线、相关性热力图
- 合成因子对比

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行

```bash
# 模拟数据模式（离线，快速测试）
python mine.py --mock

# 在线模式（需要网络和akshare）
python mine.py --start 2022-01-01 --end 2024-12-31

# 指定股票池
python mine.py --stocks 600519 000858 601318 --start 2023-01-01 --end 2024-06-30

# 指定输出路径和详细日志
python mine.py --output my_report.html --verbose
```

## 技术栈

- **数据获取**: akshare
- **数据处理**: numpy, pandas
- **统计分析**: scipy.stats
- **机器学习**: scikit-learn (PCA)
- **可视化**: matplotlib
- **报告**: HTML + 内嵌base64图片

## 因子评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| IC | Spearman(factor, forward_return) | 因子值与未来收益的秩相关系数 |
| ICIR | IC_mean / IC_std | IC的信息比率，衡量稳定性 |
| IC t值 | IC_mean / (IC_std / sqrt(n)) | IC的统计显著性 |
| IC胜率 | P(IC > 0) | IC为正的比例 |
| 衰减率 | (IC_1d - IC_n) / IC_1d | IC随时间衰减的速度 |

## 许可证

MIT License
