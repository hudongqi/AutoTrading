"""
F: PEPE stop_atr 扫描（take_R 已更新为 4.0R）
sweep stop_atr: 1.5 ~ 8.0
全程 2024-01-01 → 2026-03-24，PEPE 单币 $10,000
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
PEPE_PARAMS = {
    "atr_period": 14, "bb_period": 20, "bb_std": 2.0,
    "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60,
    "reclaim_ema": 20, "trend_ema": 200,
    "vol_period": 20, "vol_spike_mult": 1.5,
    "atr_pct_low": 0.008, "atr_pct_high": 0.20,
    "oversold_lookback": 3, "allow_short": True,
}
STOP_ATRS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]

def run(df, stop_atr):
    strat  = SOLReversionV2Strategy1H(**PEPE_PARAMS)
    df_sig = strat.generate_signals(df)
    btp    = get_sol_backtest_profile("pepe_rev_v2")
    port   = PerpPortfolio(
        initial_cash=TOTAL_CASH, leverage=btp["leverage"],
        taker_fee_rate=config.TAKER_FEE_RATE, maker_fee_rate=config.MAKER_FEE_RATE,
        maint_margin_rate=0.005,
    )
    bt = Backtester(
        broker=SimBroker(slippage_bps=config.SLIPPAGE_BPS),
        portfolio=port, strategy=strat,
        max_pos=5_000_000, cooldown_bars=btp["cooldown_bars"],
        stop_atr=stop_atr, take_R=btp["take_R"],
        trail_start_R=0.0, trail_atr=0.0, use_trailing=False,
        check_liq=True, entry_is_maker=False,
        funding_rate_per_8h=config.FUNDING_RATE_PER_8H,
        risk_per_trade=btp["risk_per_trade"],
        enable_risk_position_sizing=True,
        allow_reentry=False, partial_take_R=0.0, partial_take_frac=0.0,
        break_even_after_partial=False, break_even_R=0.0,
        use_signal_exit_targets=False, max_hold_bars=btp["max_hold_bars"],
        dd_stop_pct=0.0, dd_cooldown_bars=0,
    )
    result = bt.run(df_sig)
    stats  = result.attrs.get("stats", {})
    eq     = result["equity"]
    dd     = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret    = (eq.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    wr     = stats.get("win_rate", 0) * 100
    pf     = stats.get("profit_factor", 0)
    nc     = len(bt.closed_trades)
    calmar = ret / abs(dd) if dd != 0 else 0
    # 多空拆分
    longs  = [t for t in bt.closed_trades if t.get("side") == 1]
    shorts = [t for t in bt.closed_trades if t.get("side") == -1]
    return dict(ret=ret, dd=dd, calmar=calmar, wr=wr, pf=pf, nc=nc,
                nl=len(longs), ns=len(shorts))

print("拉取 PEPE 数据...")
ds = CCXTDataSource()
df = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"PEPE {len(df)} 根\n")

results = []
for sa in STOP_ATRS:
    print(f"  stop_atr={sa}...", end="\r", flush=True)
    results.append((sa, run(df, sa)))

print(f"\n  扫描完成\n")

W = 85
print("=" * W)
print(f"  PEPE stop_atr 扫描  |  take_R=4.0R  |  risk=4%  |  leverage=2x")
print("=" * W)
print(f"  {'stop_atr':>9} │ {'收益':>8} {'DD':>7} {'Calmar':>7} │ "
      f"{'胜率':>5} {'PF':>6} │ {'成交':>5} {'多头':>5} {'空头':>5} │ {'vs基线':>7}")
print("  " + "─" * (W - 2))

base = next(r for sa, r in results if sa == 5.0)
best = max(results, key=lambda x: x[1]["calmar"])

for sa, r in results:
    is_base = sa == 5.0
    is_best = sa == best[0]
    diff = r["ret"] - base["ret"]
    pf_s = f"{r['pf']:.2f}" if r["pf"] not in (float("inf"), 0) else "∞"
    mark = " ★" if is_base else (" ◀" if is_best else "")
    print(f"  {sa:>9.1f} │ {r['ret']:>+7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.2f} │ "
          f"{r['wr']:>4.0f}% {pf_s:>6} │ {r['nc']:>4}笔 {r['nl']:>4}多 {r['ns']:>4}空 │ "
          f"{diff:>+6.1f}%{mark}")

print(f"\n  ★ 基线 stop_atr=5.0   最优 stop_atr={best[0]}")
print(f"  基线：收益 {base['ret']:+.1f}%  DD {base['dd']:.1f}%  Calmar {base['calmar']:.2f}")
print(f"  最优：收益 {best[1]['ret']:+.1f}%  DD {best[1]['dd']:.1f}%  Calmar {best[1]['calmar']:.2f}")
print("=" * W)
