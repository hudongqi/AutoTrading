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

    portfolio = PerpPortfolio(INITIAL_CASH)
    broker = SimBroker(FEE_RATE, SLIPPAGE_BPS)

    bt = Backtester(portfolio, broker, max_pos=1)

    result = bt.run(df_sig)

    print(result.tail())

    total_return = result["equity"].iloc[-1] / INITIAL_CASH - 1

    max_dd = (result["equity"] / result["equity"].cummax() - 1).min()

    print("\n==== Result ====")
    print("Return:", f"{total_return:.2%}")
    print("Max DD:", f"{max_dd:.2%}")


if __name__ == "__main__":
    main()
