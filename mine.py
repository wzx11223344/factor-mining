"""
mine.py - Alpha因子挖掘引擎 CLI入口
======================================

命令行工具，用于执行完整的因子挖掘流程:
    1. 数据获取 (akshare / 模拟模式)
    2. Alpha因子计算 (14个因子)
    3. IC分析 (Spearman IC / ICIR / 分位数组合)
    4. 因子合成 (等权 / IC加权 / PCA / 最大化ICIR)
    5. 因子衰减分析 (1-20日)
    6. HTML报告生成

用法:
    python mine.py --start 2022-01-01 --end 2024-12-31 --output report.html
    python mine.py --mock  # 使用模拟数据模式
    python mine.py --stocks 600519 000858 --start 2023-01-01 --end 2024-06-30
"""

import argparse
import logging
import sys
from typing import List, Optional

import pandas as pd

from factor_mining.data import DataFetcher
from factor_mining.factors import FactorCalculator, FACTOR_NAMES, FACTOR_CATEGORIES
from factor_mining.ic_analysis import ICAnalyzer
from factor_mining.synthesis import FactorSynthesizer
from factor_mining.decay import DecayAnalyzer
from factor_mining.report import ReportGenerator


def setup_logging(verbose: bool = False) -> None:
    """配置日志输出。

    Args:
        verbose: 是否输出详细日志
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_factor_mining(
    stock_pool: Optional[List[str]] = None,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    mock_mode: bool = False,
    output_path: str = "factor_report.html",
    max_decay_lag: int = 20,
) -> None:
    """执行完整的因子挖掘流程。

    Args:
        stock_pool: 股票代码列表
        start_date: 数据起始日期
        end_date: 数据结束日期
        mock_mode: 是否使用模拟数据
        output_path: 报告输出路径
        max_decay_lag: 衰减分析最大滞后期
    """
    logger = logging.getLogger("mine")

    # ============ Step 1: 数据获取 ============
    logger.info("=" * 60)
    logger.info("Step 1: 数据获取")
    logger.info("=" * 60)

    fetcher = DataFetcher(
        stock_pool=stock_pool,
        start_date=start_date,
        end_date=end_date,
        mock_mode=mock_mode,
    )

    panel_data = fetcher.fetch_panel_data()
    if len(panel_data) == 0:
        logger.error("未获取到任何数据，请检查网络或使用--mock模式")
        return

    market_index = fetcher.fetch_market_index()

    # 获取市值和估值数据
    market_caps = {}
    valuations = {}
    for symbol in panel_data["symbol"].unique():
        market_caps[symbol] = fetcher.fetch_market_cap(symbol)
        valuations[symbol] = fetcher.fetch_valuation_data(symbol)

    logger.info(f"数据获取完成: {len(panel_data)}条记录, {panel_data['symbol'].nunique()}只股票")

    # ============ Step 2: 因子计算 ============
    logger.info("=" * 60)
    logger.info("Step 2: Alpha因子计算")
    logger.info("=" * 60)

    calculator = FactorCalculator(
        panel_data=panel_data,
        market_index=market_index,
        market_caps=market_caps,
        valuations=valuations,
    )

    factor_df = calculator.calculate_all_factors()
    available_factors = [f for f in FACTOR_NAMES if f in factor_df.columns]
    logger.info(f"因子计算完成: {len(available_factors)}/{len(FACTOR_NAMES)}个因子可用")

    # ============ Step 3: IC分析 ============
    logger.info("=" * 60)
    logger.info("Step 3: IC分析")
    logger.info("=" * 60)

    ic_analyzer = ICAnalyzer(factor_df, forward_days=1)
    ic_summary_df, quantile_summary_df = ic_analyzer.analyze_all_factors(available_factors)
    ic_matrix = ic_analyzer.get_ic_matrix(available_factors)
    factor_corr = ic_analyzer.compute_factor_correlation(available_factors)

    logger.info("IC分析完成")
    logger.info(f"\n{ic_summary_df.to_string(index=False)}")

    # ============ Step 4: 因子合成 ============
    logger.info("=" * 60)
    logger.info("Step 4: 因子合成")
    logger.info("=" * 60)

    synthesizer = FactorSynthesizer(factor_df, available_factors, forward_days=1)
    synth_df, synth_ic_df = synthesizer.synthesize_all()

    # 合并单因子和合成因子IC对比
    synth_comparison = pd.concat([ic_summary_df, synth_ic_df], ignore_index=True)
    synth_comparison = synth_comparison.sort_values("icir", ascending=False)

    logger.info("因子合成完成")
    logger.info(f"\n{synth_comparison.to_string(index=False)}")

    # ============ Step 5: 因子衰减分析 ============
    logger.info("=" * 60)
    logger.info("Step 5: 因子衰减分析")
    logger.info("=" * 60)

    decay_analyzer = DecayAnalyzer(factor_df, max_lag=max_decay_lag)
    decay_summary, best_periods = decay_analyzer.analyze_all_factors(available_factors)

    logger.info("衰减分析完成")
    logger.info(f"\n{best_periods.to_string(index=False)}")

    # ============ Step 6: 生成报告 ============
    logger.info("=" * 60)
    logger.info("Step 6: HTML报告生成")
    logger.info("=" * 60)

    report_gen = ReportGenerator(output_path=output_path)
    report_gen.generate_report(
        ic_summary=ic_summary_df,
        quantile_summary=quantile_summary_df,
        ic_matrix=ic_matrix,
        decay_summary=decay_summary,
        best_periods=best_periods,
        synth_comparison=synth_comparison,
        factor_corr=factor_corr,
        factor_categories=FACTOR_CATEGORIES,
    )

    logger.info(f"\n报告已生成: {output_path}")
    logger.info("因子挖掘流程完成!")


def main() -> None:
    """CLI主函数。"""
    parser = argparse.ArgumentParser(
        description="Alpha因子挖掘引擎 - A股因子分析与IC评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python mine.py --mock                                    # 模拟数据快速运行
  python mine.py --start 2022-01-01 --end 2024-12-31       # 指定日期范围
  python mine.py --stocks 600519 000858 601318            # 指定股票池
  python mine.py --output my_report.html --verbose        # 指定输出和详细日志
        """,
    )

    parser.add_argument(
        "--stocks", "-s",
        nargs="+",
        default=None,
        help="股票代码列表 (如: 600519 000858)，不指定则使用默认股票池",
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        help="数据起始日期 (YYYY-MM-DD)，默认 2022-01-01",
    )
    parser.add_argument(
        "--end",
        default="2024-12-31",
        help="数据结束日期 (YYYY-MM-DD)，默认 2024-12-31",
    )
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="使用模拟数据模式 (离线环境)",
    )
    parser.add_argument(
        "--output", "-o",
        default="factor_report.html",
        help="HTML报告输出路径，默认 factor_report.html",
    )
    parser.add_argument(
        "--max-decay-lag",
        type=int,
        default=20,
        help="衰减分析最大滞后期 (交易日)，默认 20",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    try:
        run_factor_mining(
            stock_pool=args.stocks,
            start_date=args.start,
            end_date=args.end,
            mock_mode=args.mock,
            output_path=args.output,
            max_decay_lag=args.max_decay_lag,
        )
    except KeyboardInterrupt:
        print("\n用户中断执行")
        sys.exit(1)
    except Exception as e:
        logging.error(f"执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
