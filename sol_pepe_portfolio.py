"""
SOL + PEPE 双币种组合回测
各分配 50% 资金（5,000 USDT），策略参数独立，同步运行
对比：单独 SOL / 单独 PEPE / 组合表现
测试区间：2024-01-01 → 2026-03-24
"""

import pandas as pd
import numpy as np

from data import CCXTDataSource
from strategy import SOLReversionV2Strategy1H
from strategy_profiles import get_sol_backtest_profile
from backtest import Backtester
from portfolio import PerpPortfolio
from broker import SimBroker
import config

TOTAL_CASH = 10_000
HALF_CASH  = TOTAL_CASH // 2   # 5,000 每个币种

PERIODS = [
    ("2024-01-01", "2025-01-01", "2024全年"),
    ("2025-01-01", "2026-03-24", "2025-26"),
    ("2024-01-01", "2026-03-24", "2024-26总计"),
]

# ── 策略参数 ──────────────────────────────────────────────────
SOL_PARAMS = {
    "atr_period": 14, "bb_period": 20, "bb_std": 2.0,
    "rsi_period": 14, "rsi_oversold": 38, "rsi_overbought": 62,
    "reclaim_ema": 20, "trend_ema": 200,
    "vol_period": 20, "vol_spike_mult": 1.2,
    "atr_pct_low": 0.003, "atr_pct_high": 0.10,
    "oversold_lookback": 3, "allow_short": True,
}

PEPE_PARAMS = {
    **SOL_PARAMS,
    "rsi_oversold": 40, "rsi_overbought": 60,  # PEPE 保持原有 RSI（优化扫描未改动）
    "vol_spike_mult": 1.5,   # PEPE 专用：过滤弱量能假突破
    "atr_pct_low": 0.008,    # 第二轮优化：过滤低波动期，Calmar +7%
    "atr_pct_high": 0.20,    # PEPE 高波动需要放宽上限
}


def run_single_coin(symbol, strat_params, cash, start, end, max_pos=None, bt_profile="sol_rev_v2"):
    """运行单个币种回测，返回权益曲线和统计数据"""
    ds  = CCXTDataSource()
    df  = ds.load_ohlcv(symbol=symbol, start=start, end=end, timeframe="1h")

    strat  = SOLReversionV2Strategy1H(**strat_params)
    df_sig = strat.generate_signals(df)

    btp = get_sol_backtest_profile(bt_profile)
    portfolio = PerpPortfolio(
        initial_cash      = cash,
        leverage          = btp["leverage"],
        taker_fee_rate    = config.TAKER_FEE_RATE,
        maker_fee_rate    = config.MAKER_FEE_RATE,
        maint_margin_rate = 0.005,
    )
    broker = SimBroker(slippage_bps=config.SLIPPAGE_BPS)
    bt = Backtester(
        broker=broker, portfolio=portfolio, strategy=strat,
        max_pos           = max_pos if max_pos else btp["max_pos"],
        cooldown_bars     = btp["cooldown_bars"],
        stop_atr          = btp["stop_atr"],
        take_R            = btp["take_R"],
        trail_start_R     = btp["trail_start_R"],
        trail_atr         = btp["trail_atr"],
        use_trailing      = btp["use_trailing"],
        check_liq         = True,
        entry_is_maker    = btp["entry_is_maker"],
        funding_rate_per_8h = config.FUNDING_RATE_PER_8H,
        risk_per_trade    = btp["risk_per_trade"],
        enable_risk_position_sizing = btp["enable_risk_position_sizing"],
        allow_reentry     = btp["allow_reentry"],
        partial_take_R    = btp["partial_take_R"],
        partial_take_frac = btp["partial_take_frac"],
        break_even_after_partial = btp["break_even_after_partial"],
        break_even_R      = btp["break_even_R"],
        use_signal_exit_targets = btp["use_signal_exit_targets"],
        max_hold_bars     = btp["max_hold_bars"],
    )

    result = bt.run(df_sig)
    stats  = result.attrs.get("stats", {})
    closed = result.attrs.get("closed_trades", [])

    coin_return = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
    n_long  = int((df_sig["entry_setup"] ==  1).sum())
    n_short = int((df_sig["entry_setup"] == -1).sum())

    return result["equity"], stats, closed, coin_return, n_long, n_short


def calc_metrics(equity_series, initial_cash):
    final    = equity_series.iloc[-1]
    peak     = equity_series.cummax()
    dd_series = (equity_series - peak) / peak
    max_dd   = dd_series.min() * 100
    ret      = (final - initial_cash) / initial_cash * 100

    # 简单 Sharpe（用每日收益近似）
    daily    = equity_series.resample("1D").last().pct_change().dropna()
    sharpe   = (daily.mean() / daily.std() * np.sqrt(365)).round(2) if daily.std() > 0 else 0

    return dict(ret=ret, max_dd=max_dd, final=final, sharpe=sharpe)


def print_sep(label):
    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"{'─'*65}")


# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("  SOL + PEPE 双币种组合回测")
print(f"  总资金 {TOTAL_CASH:,} USDT，各分配 {HALF_CASH:,} USDT（50/50）")
print("=" * 65)

for start, end, period_label in PERIODS:
    print_sep(f"{period_label}  [{start} → {end}]")

    # ── SOL ──────────────────────────────────────────────────
    print(f"\n  拉取 SOL 数据...")
    sol_eq, sol_stats, sol_closed, sol_coin, sol_nl, sol_ns = run_single_coin(
        "SOL/USDT:USDT", SOL_PARAMS, HALF_CASH, start, end
    )
    sol_m = calc_metrics(sol_eq, HALF_CASH)

    # ── PEPE ─────────────────────────────────────────────────
    print(f"  拉取 PEPE 数据...")
    pep_eq, pep_stats, pep_closed, pep_coin, pep_nl, pep_ns = run_single_coin(
        "1000PEPE/USDT:USDT", PEPE_PARAMS, HALF_CASH, start, end,
        max_pos=5_000_000, bt_profile="pepe_rev_v2"
    )
    pep_m = calc_metrics(pep_eq, HALF_CASH)

    # ── 合并权益曲线 ─────────────────────────────────────────
    # 将两条曲线对齐到相同时间轴，缺失时用前值填充
    combined = (
        sol_eq.reindex(sol_eq.index.union(pep_eq.index)).ffill()
        + pep_eq.reindex(sol_eq.index.union(pep_eq.index)).ffill()
    )
    comb_m = calc_metrics(combined, TOTAL_CASH)

    # 相关性（日收益）
    sol_daily  = sol_eq.resample("1D").last().pct_change().dropna()
    pep_daily  = pep_eq.resample("1D").last().pct_change().dropna()
    common_idx = sol_daily.index.intersection(pep_daily.index)
    corr = sol_daily.loc[common_idx].corr(pep_daily.loc[common_idx]) if len(common_idx) > 5 else float("nan")

    # ── 打印结果 ─────────────────────────────────────────────
    def pf_str(stats):
        pf = stats.get("profit_factor", 0)
        return "∞" if pf == float("inf") else f"{pf:.2f}"

    print(f"\n  {'指标':<16} {'SOL ($5k)':>12} {'PEPE ($5k)':>12} {'组合 ($10k)':>12}")
    print(f"  {'─'*54}")
    print(f"  {'收益率':<16} {sol_m['ret']:>+11.2f}%  {pep_m['ret']:>+11.2f}%  {comb_m['ret']:>+11.2f}%")
    print(f"  {'最终资金':<16} {sol_m['final']:>11,.0f}  {pep_m['final']:>11,.0f}  {comb_m['final']:>11,.0f}")
    print(f"  {'最大回撤':<16} {sol_m['max_dd']:>11.2f}%  {pep_m['max_dd']:>11.2f}%  {comb_m['max_dd']:>11.2f}%")
    print(f"  {'Sharpe':<16} {sol_m['sharpe']:>12.2f}  {pep_m['sharpe']:>12.2f}  {comb_m['sharpe']:>12.2f}")
    print(f"  {'成交笔数':<16} {sol_stats.get('closed_trade_count', len(sol_closed)):>12}  "
          f"{pep_stats.get('closed_trade_count', len(pep_closed)):>12}")
    print(f"  {'胜率':<16} {sol_stats.get('win_rate',0)*100:>11.1f}%  {pep_stats.get('win_rate',0)*100:>11.1f}%")
    print(f"  {'利润因子':<16} {pf_str(sol_stats):>12}  {pf_str(pep_stats):>12}")
    print(f"  {'信号[多/空]':<16} [{sol_nl}/{sol_ns}]{'':>6}   [{pep_nl}/{pep_ns}]")
    print(f"  {'标的同期涨跌':<16} {sol_coin:>+11.2f}%  {pep_coin:>+11.2f}%")
    print(f"\n  两币种日收益相关性: {corr:.3f}  （越低 = 组合分散效果越好）")

    # 出场分布
    sol_es = sol_stats.get("exit_reason_split", {})
    pep_es = pep_stats.get("exit_reason_split", {})
    sol_es_str = "  ".join(f"{k}:{v}" for k,v in sorted(sol_es.items()))
    pep_es_str = "  ".join(f"{k}:{v}" for k,v in sorted(pep_es.items()))
    print(f"  SOL 出场:  {sol_es_str}")
    print(f"  PEPE 出场: {pep_es_str}")

print("\n" + "=" * 65)
print("  完成")
print("=" * 65)
