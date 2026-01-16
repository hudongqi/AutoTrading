import pandas as pd
import numpy as np

class Backtester:
    """
    对接 PerpPortfolio 的永续回测器：
    - 成交与记账：portfolio.apply_fill(fill_price, qty)
    - 估值：portfolio.equity(mark_price)
    - 保证金：portfolio.margin_used / free_margin
    - 止损止盈：用 self.cur_stop/self.cur_take（持仓状态）
    """

    def __init__(
        self,
        portfolio,          # PerpPortfolio
        broker,             # SimBroker（仅滑点）
        max_pos: float = 1.0,   # 目标仓位（BTC 数量），比如 0.1 / 1.0
        stop_atr: float = 2.0,
        take_atr: float = 3.0,
        use_trailing: bool = True,
        check_liq: bool = True,  # 是否做简化爆仓检查
    ):
        self.portfolio = portfolio
        self.broker = broker
        self.max_pos = float(max_pos)

        self.stop_atr = float(stop_atr)
        self.take_atr = float(take_atr)
        self.use_trailing = bool(use_trailing)
        self.check_liq = bool(check_liq)

        self.cur_stop = None
        self.cur_take = None

    def _set_brackets(self, entry_price: float, atr: float, side: int):
        if atr is None or np.isnan(atr) or atr <= 0:
            self.cur_stop, self.cur_take = None, None
            return

        if side == 1:
            self.cur_stop = entry_price - self.stop_atr * atr
            self.cur_take = entry_price + self.take_atr * atr
        else:
            self.cur_stop = entry_price + self.stop_atr * atr
            self.cur_take = entry_price - self.take_atr * atr

    def _update_trailing_stop(self, close: float, atr: float, side: int):
        if not self.use_trailing:
            return
        if self.cur_stop is None:
            return
        if atr is None or np.isnan(atr) or atr <= 0:
            return

        if side == 1:
            # 多头：止损只上移
            new_stop = close - self.stop_atr * atr
            self.cur_stop = max(self.cur_stop, new_stop)
        else:
            # 空头：止损只下移（数值更小更有利）
            new_stop = close + self.stop_atr * atr
            self.cur_stop = min(self.cur_stop, new_stop)

    def _close_position(self, exit_mid_price: float, reason: str) -> dict:
        """用 portfolio.apply_fill 平仓，并返回事件信息。"""
        st = self.portfolio.state
        if st.position == 0:
            return {"exit_reason": None, "exit_price": np.nan}

        qty_to_close = -st.position
        fill = self.broker.fill_price(exit_mid_price, qty_to_close)

        # 记账（手续费 + realized pnl 由 portfolio 处理）
        self.portfolio.apply_fill(fill_price=fill, qty=qty_to_close, is_maker=False)

        # 平仓后清空风控线
        self.cur_stop, self.cur_take = None, None

        return {"exit_reason": reason, "exit_price": float(fill)}

    def run(self, df_signals: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for ts, row in df_signals.iterrows():
            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            atr = float(row["atr"]) if ("atr" in row and pd.notna(row["atr"])) else np.nan

            st = self.portfolio.state
            pos = float(st.position)

            exit_reason = None
            exit_price = np.nan

            # ========== (可选) 简化爆仓检查 ==========
            if self.check_liq and self.portfolio.is_liquidation_risk(close):
                evt = self._close_position(exit_mid_price=close, reason="LIQ")
                exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                pos = float(self.portfolio.state.position)

            # ========== A) 先检查止损/止盈 ==========
            if pos != 0 and (self.cur_stop is not None) and (self.cur_take is not None):
                side = 1 if pos > 0 else -1

                if side == 1:
                    hit_stop = low <= self.cur_stop
                    hit_take = high >= self.cur_take

                    # 同根同时触发：保守先止损
                    if hit_stop:
                        evt = self._close_position(exit_mid_price=self.cur_stop, reason="STOP")
                        exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                    elif hit_take:
                        evt = self._close_position(exit_mid_price=self.cur_take, reason="TAKE")
                        exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]

                else:
                    hit_stop = high >= self.cur_stop
                    hit_take = low <= self.cur_take

                    if hit_stop:
                        evt = self._close_position(exit_mid_price=self.cur_stop, reason="STOP")
                        exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                    elif hit_take:
                        evt = self._close_position(exit_mid_price=self.cur_take, reason="TAKE")
                        exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]

            # 重新读取持仓（因为可能刚刚止损/止盈/爆仓平了）
            st = self.portfolio.state
            pos = float(st.position)

            # ========== B) 再处理交易信号（事件驱动） ==========
            trade_sig = float(row["trade_signal"])
            signal = int(row["signal"])

            if trade_sig != 0:
                # 目标仓位（BTC 数量）
                target = 0.0
                if signal == 1:
                    target = self.max_pos
                elif signal == -1:
                    target = -self.max_pos

                # 由 portfolio 做限仓（保证金约束），返回需要成交的 qty
                order_qty = float(self.portfolio.target_position(target, close))

                if order_qty != 0:
                    fill = self.broker.fill_price(close, order_qty)
                    self.portfolio.apply_fill(fill_price=fill, qty=order_qty, is_maker=False)

                    # 成交后如果有仓位，设置止损止盈（以成交价为基准）
                    st = self.portfolio.state
                    if st.position != 0:
                        side = 1 if st.position > 0 else -1
                        self._set_brackets(entry_price=float(fill), atr=atr, side=side)
                    else:
                        self.cur_stop, self.cur_take = None, None

            # ========== C) 持仓期间追踪止损 ==========
            st = self.portfolio.state
            if st.position != 0:
                side = 1 if st.position > 0 else -1
                self._update_trailing_stop(close=close, atr=atr, side=side)

            # ========== D) 记录快照 ==========
            st = self.portfolio.state
            eq = self.portfolio.equity(close)
            upnl = self.portfolio.unrealized_pnl(close)
            m_used = self.portfolio.margin_used(close)
            f_margin = self.portfolio.free_margin(close)

            rows.append({
                "time": ts,
                "close": close,
                "high": high,
                "low": low,
                "atr": atr,

                "cash": st.cash,
                "position": st.position,
                "avg_price": st.avg_price,
                "realized_pnl": st.realized_pnl,
                "unrealized_pnl": upnl,
                "equity": eq,

                "margin_used": m_used,
                "free_margin": f_margin,

                "stop_price": self.cur_stop,
                "take_price": self.cur_take,

                "signal": signal,
                "trade_signal": trade_sig,

                "exit_reason": exit_reason,
                "exit_price": exit_price,
                "liq_risk": self.portfolio.is_liquidation_risk(close),
            })

        out = pd.DataFrame(rows).set_index("time")
        out["returns"] = out["equity"].pct_change().fillna(0)
        out["cum_returns"] = (1 + out["returns"]).cumprod()
        return out
