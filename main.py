import pandas as pd

from config import *
from data import CCXTDataSource
from strategy import BTCPerpTrendStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester


def main():

    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    strat = BTCPerpTrendStrategy1H(fast=5, slow=15)
    df_sig = strat.generate_signals(df)

    portfolio = PerpPortfolio(
        initial_cash=INITIAL_CASH,
        leverage=2,
        taker_fee_rate=FEE_RATE,
        maker_fee_rate=FEE_RATE,  # 不区分就填一样
        maint_margin_rate=0.005
    )

    broker = SimBroker(slippage_bps=SLIPPAGE_BPS)

    bt = Backtester(
        broker=broker,
        portfolio=portfolio,
        strategy=strat,
        max_pos=0.1,
        cooldown_bars=3,
        stop_atr=2.0,
        take_R=4.0,
        trail_start_R=1.0,
        trail_atr=1.5,
        use_trailing=True,
        check_liq=True
    )

    result = bt.run(df_sig)
    print(result[["close", "position", "equity", "margin_used", "free_margin", "exit_reason"]].tail(20))

    stats = result.attrs.get("stats", {})
    print("\n==== Trade Stats ====")
    print("交易次数:", stats.get("trade_count", 0))
    print("手续费总额:", f"{stats.get('total_fees', 0.0):.4f}")
    print("反手次数:", stats.get("reversal_count", 0))
    print("胜率:", f"{stats.get('win_rate', 0.0):.2%}")

    pnl_ratio = stats.get("pnl_ratio", float("nan"))
    if pd.notna(pnl_ratio):
        print("盈亏比:", f"{pnl_ratio:.2f}")
    else:
        print("盈亏比: N/A（样本不足）")


if __name__ == "__main__":
    main()
