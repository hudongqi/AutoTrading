import pandas as pd

from config import *
from data import CCXTDataSource
from strategy import BTCPerpTrendStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester


def run_case(
    name: str,
    df_sig,
    strat,
    entry_is_maker=False,
    funding_rate_per_8h=0.0,
    leverage=2.0,
    max_pos=0.8,
    cooldown_bars=3,
    stop_atr=1.5,
    take_R=3.5,
    trail_start_R=1.5,
    trail_atr=2.0,
    show_result_tail=False,
    debug_breakpoint=False,
):
    portfolio = PerpPortfolio(
        initial_cash=INITIAL_CASH,
        leverage=leverage,
        taker_fee_rate=TAKER_FEE_RATE,
        maker_fee_rate=MAKER_FEE_RATE,
        maint_margin_rate=0.005
    )
    broker = SimBroker(slippage_bps=SLIPPAGE_BPS)

    bt = Backtester(
        broker=broker,
        portfolio=portfolio,
        strategy=strat,
        max_pos=max_pos,
        cooldown_bars=cooldown_bars,
        stop_atr=stop_atr,
        take_R=take_R,
        trail_start_R=trail_start_R,
        trail_atr=trail_atr,
        use_trailing=True,
        check_liq=True,
        entry_is_maker=entry_is_maker,
        funding_rate_per_8h=funding_rate_per_8h,
    )

    result = bt.run(df_sig)
    stats = result.attrs.get("stats", {})
    eq = result["equity"]
    max_dd = (eq / eq.cummax() - 1).min()

    if show_result_tail:
        print(result[["close", "position", "equity", "margin_used", "free_margin", "exit_reason"]].tail(20))

    if debug_breakpoint:
        breakpoint()

    print(f"\n==== {name} ====")
    print("final_equity:", f"{result['equity'].iloc[-1]:.4f}")
    print("return:", f"{(result['equity'].iloc[-1] / INITIAL_CASH - 1):.2%}")
    print("max_drawdown:", f"{max_dd:.2%}")
    print("fees:", f"{stats.get('total_fees', 0.0):.4f}")
    print("funding_total:", f"{stats.get('funding_total', 0.0):.4f}")
    print("win_rate:", f"{stats.get('win_rate', 0.0):.2%}")

    pnl_ratio = stats.get("pnl_ratio", float("nan"))
    if pd.notna(pnl_ratio):
        print("pnl_ratio:", f"{pnl_ratio:.2f}")
    else:
        print("pnl_ratio: N/A")


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    # BASELINE_REAL_FEE_V1: 真实手续费，开/平仓统一按 taker
    strat_base = BTCPerpTrendStrategy1H(fast=5, slow=15)
    df_sig_base = strat_base.generate_signals(df)
    run_case("BASELINE_REAL_FEE_V1", df_sig_base, strat_base, entry_is_maker=False, funding_rate_per_8h=0.0)

    # maker-entry 测试：开仓按 maker，平仓继续按 taker
    run_case("MAKER_ENTRY_TEST_V1", df_sig_base, strat_base, entry_is_maker=True, funding_rate_per_8h=0.0)

    # funding 模拟：在 baseline 基础上加入资金费
    run_case("BASELINE_REAL_FEE_V1_WITH_FUNDING", df_sig_base, strat_base, entry_is_maker=False, funding_rate_per_8h=FUNDING_RATE_PER_8H)

    # AGGRESSIVE_GROWTH_V1（大胆版本）：目标高收益且回撤控制在 30% 内
    strat_aggr = BTCPerpTrendStrategy1H(
        fast=5,
        slow=15,
        atr_pct_threshold=0.0038,
        use_regime_filter=True,
        adx_threshold_4h=32,
        trend_strength_threshold_4h=0.006,
        slow_slope_lookback_4h=3,
    )
    df_sig_aggr = strat_aggr.generate_signals(df)
    run_case(
        "AGGRESSIVE_GROWTH_V1",
        df_sig_aggr,
        strat_aggr,
        entry_is_maker=True,
        funding_rate_per_8h=0.0,
        leverage=3.6,
        max_pos=0.8,
        cooldown_bars=3,
        stop_atr=1.3,
        take_R=2.7,
        trail_start_R=0.8,
        trail_atr=3.0,
        show_result_tail=True,
        debug_breakpoint=True,
    )


if __name__ == "__main__":
    main()
