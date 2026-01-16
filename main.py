from config import *
from data import CCXTDataSource
from strategy import BTCPerpTrendStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester


def main():

    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    strat = BTCPerpTrendStrategy1H(fast=10, slow=30)
    df_sig = strat.generate_signals(df)

    portfolio = PerpPortfolio(
        initial_cash=INITIAL_CASH,
        leverage=10,
        taker_fee_rate=FEE_RATE,
        maker_fee_rate=FEE_RATE,  # 不区分就填一样
        maint_margin_rate=0.005
    )

    broker = SimBroker(slippage_bps=SLIPPAGE_BPS)

    bt = Backtester(
        portfolio=portfolio,
        broker=broker,
        max_pos=0.1,  # 比如 0.1 BTC
        stop_atr=2.0,
        take_atr=3.0,
        use_trailing=True,
        check_liq=True
    )

    result = bt.run(df_sig)
    print(result[["close", "position", "equity", "margin_used", "free_margin", "exit_reason"]].tail(20))

    # total_return = result["equity"].iloc[-1] / INITIAL_CASH - 1
    #
    # max_dd = (result["equity"] / result["equity"].cummax() - 1).min()
    #
    # print("\n==== Result ====")
    # print("Return:", f"{total_return:.2%}")
    # print("Max DD:", f"{max_dd:.2%}")


if __name__ == "__main__":
    main()
