import pandas as pd

from config import *
from data import CCXTDataSource
from strategy import BTCPerpTrendStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester


def run_case(name: str, df_sig, strat, entry_is_maker=False, funding_rate_per_8h=0.0):
    portfolio = PerpPortfolio(
        initial_cash=INITIAL_CASH,
        leverage=2,
        taker_fee_rate=TAKER_FEE_RATE,
        maker_fee_rate=MAKER_FEE_RATE,
        maint_margin_rate=0.005
    )
    broker = SimBroker(slippage_bps=SLIPPAGE_BPS)

    bt = Backtester(
        broker=broker,
        portfolio=portfolio,
        strategy=strat,
        max_pos=0.8,
        cooldown_bars=3,
        stop_atr=1.5,
        take_R=3.5,
        trail_start_R=1.5,
        trail_atr=2.0,
        use_trailing=True,
        check_liq=True,
        entry_is_maker=entry_is_maker,
        funding_rate_per_8h=funding_rate_per_8h,
    )

    result = bt.run(df_sig)
    stats = result.attrs.get("stats", {})
    print(f"\n==== {name} ====")
    print("final_equity:", f"{result['equity'].iloc[-1]:.4f}")
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

    strat = BTCPerpTrendStrategy1H(fast=5, slow=15)
    df_sig = strat.generate_signals(df)

    # BASELINE_REAL_FEE_V1: 真实手续费，开/平仓统一按 taker
    run_case("BASELINE_REAL_FEE_V1", df_sig, strat, entry_is_maker=False, funding_rate_per_8h=0.0)

    # maker-entry 测试：开仓按 maker，平仓继续按 taker
    run_case("MAKER_ENTRY_TEST_V1", df_sig, strat, entry_is_maker=True, funding_rate_per_8h=0.0)

    # funding 模拟：在 baseline 基础上加入资金费
    run_case("BASELINE_REAL_FEE_V1_WITH_FUNDING", df_sig, strat, entry_is_maker=False, funding_rate_per_8h=FUNDING_RATE_PER_8H)


if __name__ == "__main__":
    main()
