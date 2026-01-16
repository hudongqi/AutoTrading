import pandas as pd
import numpy as np

class Backtester:

    def __init__(self, portfolio, broker, max_pos=100,
                 stop_atr=2, take_atr=3, use_trailing=True):

        self.portfolio = portfolio
        self.broker = broker
        self.max_pos = max_pos

        self.stop_atr = stop_atr
        self.take_atr = take_atr
        self.use_trailing = use_trailing

        self.cur_stop = None
        self.cur_take = None

    def _set_brackets(self, entry, atr, side):

        if atr <= 0 or np.isnan(atr):
            return

        if side == 1:
            self.cur_stop = entry - atr * self.stop_atr
            self.cur_take = entry + atr * self.take_atr
        else:
            self.cur_stop = entry + atr * self.stop_atr
            self.cur_take = entry - atr * self.take_atr

    def run(self, df):

        rows = []

        st = self.portfolio.state

        for t, row in df.iterrows():

            close = row.close
            high = row.high
            low = row.low
            atr = row.atr

            # ====== 止盈止损优先 ======

            if st.position != 0 and self.cur_stop:

                side = 1 if st.position > 0 else -1

                stop_hit = low <= self.cur_stop if side == 1 else high >= self.cur_stop
                take_hit = high >= self.cur_take if side == 1 else low <= self.cur_take

                if stop_hit or take_hit:

                    exit_price = self.cur_stop if stop_hit else self.cur_take

                    qty = -st.position
                    fill = self.broker.fill_price(exit_price, qty)

                    st.cash -= fill * qty
                    st.cash -= self.broker.commission(fill * qty)

                    st.position = 0
                    st.avg_price = 0
                    self.cur_stop = None
                    self.cur_take = None

            # ====== 信号交易 ======

            if row.trade_signal != 0:

                target = 0
                if row.signal == 1:
                    target = self.max_pos
                elif row.signal == -1:
                    target = -self.max_pos

                qty = self.portfolio.target_position(target, close)

                if qty != 0:

                    fill = self.broker.fill_price(close, qty)

                    st.cash -= fill * qty
                    st.cash -= self.broker.commission(fill * qty)

                    st.position += qty
                    st.avg_price = fill

                    self._set_brackets(fill, atr, 1 if st.position > 0 else -1)

            rows.append({
                "time": t,
                "equity": self.portfolio.value(close),
                "position": st.position,
                "close": close
            })

        out = pd.DataFrame(rows).set_index("time")

        out["returns"] = out["equity"].pct_change().fillna(0)
        out["cum_returns"] = (1 + out["returns"]).cumprod()

        return out
