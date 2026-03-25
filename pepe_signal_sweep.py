"""
I: PEPE 入场信号参数扫描
rsi_oversold × vol_spike_mult × atr_pct_low
全程 2024-01-01 → 2026-03-24，PEPE 单币 $10,000
"""
import pandas as pd
import numpy as np
from itertools import product
from data import CCXTDataSource
from strategy import SOLReversionV2Strategy1H
from strategy_profiles import get_sol_backtest_profile
from backtest import Backtester
from portfolio import PerpPortfolio
from broker import SimBroker
import config

TOTAL_CASH = 10_000
BASE_PARAMS = {
    "atr_period": 14, "bb_period": 20, "bb_std": 2.0,
    "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60,
    "reclaim_ema": 20, "trend_ema": 200,
    "vol_period": 20, "vol_spike_mult": 1.5,
    "atr_pct_low": 0.008, "atr_pct_high": 0.20,
    "oversold_lookback": 3, "allow_short": True,
}

RSI_OVERSOLDS  = [35, 38, 40, 42, 45]
VOL_SPIKES     = [1.2, 1.5, 1.8, 2.0, 2.5]
ATR_PCT_LOWS   = [0.005, 0.008, 0.010, 0.015]


def run(df, params):
    strat  = SOLReversionV2Strategy1H(**params)
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
        stop_atr=btp["stop_atr"], take_R=btp["take_R"],
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
    nc     = len(bt.closed_trades)
    calmar = ret / abs(dd) if dd != 0 else 0
    return dict(ret=ret, dd=dd, calmar=calmar, wr=wr, nc=nc)


print("拉取 PEPE 数据...")
ds = CCXTDataSource()
df = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"PEPE {len(df)} 根\n")

base_r = run(df, BASE_PARAMS)

W = 88

# ── 1. rsi_oversold × vol_spike_mult ─────────────────────
print(f"{'='*W}")
print("  [1/3] rsi_oversold × vol_spike_mult（atr_pct_low=0.008 固定）")
print(f"{'='*W}")
print(f"  {'RSI':>5} {'vol_mult':>8} │ {'收益':>8} {'DD':>7} {'Calmar':>7} {'胜率':>5} {'成交':>5} │ {'vs基线':>7}")
print("  " + "─" * (W - 2))

rv_results = []
combos = list(product(RSI_OVERSOLDS, VOL_SPIKES))
for idx, (rsi, vol) in enumerate(combos):
    print(f"  [{idx+1:>2}/{len(combos)}] rsi={rsi} vol={vol}...", end="\r", flush=True)
    p = {**BASE_PARAMS, "rsi_oversold": rsi, "vol_spike_mult": vol}
    r = run(df, p)
    rv_results.append((rsi, vol, r))

rv_results.sort(key=lambda x: -x[2]["calmar"])
print(f"\n")

for rsi, vol, r in rv_results:
    is_base = (rsi == 40 and vol == 1.5)
    diff    = r["ret"] - base_r["ret"]
    mark    = " ★" if is_base else (" ◀" if r is rv_results[0][2] else "")
    print(f"  {rsi:>5}  {vol:>8.1f} │ {r['ret']:>+7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.2f} "
          f"{r['wr']:>4.0f}% {r['nc']:>4}笔 │ {diff:>+6.1f}%{mark}")

# ── 2. atr_pct_low（其他固定在基线）────────────────────────
print(f"\n{'='*W}")
print("  [2/3] atr_pct_low 单独扫描（rsi=40, vol=1.5 固定）")
print(f"{'='*W}")
print(f"  {'atr_pct_low':>12} │ {'收益':>8} {'DD':>7} {'Calmar':>7} {'胜率':>5} {'成交':>5} │ {'vs基线':>7}")
print("  " + "─" * (W - 2))

atr_results = []
for apl in ATR_PCT_LOWS:
    p = {**BASE_PARAMS, "atr_pct_low": apl}
    r = run(df, p)
    atr_results.append((apl, r))

for apl, r in atr_results:
    is_base = abs(apl - 0.008) < 1e-6
    diff    = r["ret"] - base_r["ret"]
    mark    = " ★" if is_base else (" ◀" if r["calmar"] == max(x[1]["calmar"] for x in atr_results) else "")
    print(f"  {apl:>12.3f} │ {r['ret']:>+7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.2f} "
          f"{r['wr']:>4.0f}% {r['nc']:>4}笔 │ {diff:>+6.1f}%{mark}")

# ── 3. 最优组合验证 ──────────────────────────────────────
best_rv  = rv_results[0]
best_apl = max(atr_results, key=lambda x: x[1]["calmar"])

print(f"\n{'='*W}")
print("  [3/3] 最优组合验证")
print(f"{'='*W}")

combos_final = [
    ("基线",       {**BASE_PARAMS}),
    ("最优 rsi×vol", {**BASE_PARAMS,
                     "rsi_oversold": best_rv[0],
                     "vol_spike_mult": best_rv[1]}),
    ("最优 atr_low", {**BASE_PARAMS,
                     "atr_pct_low": best_apl[0]}),
    ("全组合最优",  {**BASE_PARAMS,
                     "rsi_oversold": best_rv[0],
                     "vol_spike_mult": best_rv[1],
                     "atr_pct_low": best_apl[0]}),
]
print(f"  {'配置':<16} {'rsi':>5} {'vol':>5} {'apl':>7} │ "
      f"{'收益':>8} {'DD':>7} {'Calmar':>7} {'胜率':>5} {'成交':>5}")
print("  " + "─" * (W - 2))
for label, p in combos_final:
    r = run(df, p)
    print(f"  {label:<16} {p['rsi_oversold']:>5} {p['vol_spike_mult']:>5.1f} {p['atr_pct_low']:>7.3f} │ "
          f"{r['ret']:>+7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.2f} "
          f"{r['wr']:>4.0f}% {r['nc']:>4}笔")

print(f"\n  基线：rsi={BASE_PARAMS['rsi_oversold']}  vol={BASE_PARAMS['vol_spike_mult']}  "
      f"atr_pct_low={BASE_PARAMS['atr_pct_low']}")
print(f"  最优 rsi×vol：rsi={best_rv[0]}  vol={best_rv[1]:.1f}  Calmar {best_rv[2]['calmar']:.2f}")
print(f"  最优 atr_low：{best_apl[0]:.3f}  Calmar {best_apl[1]['calmar']:.2f}")
print("=" * W)
