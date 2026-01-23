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
        max_pos: float = 0.5,   # 目标仓位（BTC 数量），比如 0.1 / 1.0
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
        self.entry_price = None
        self.trail_start_atr = 1.5

    def _set_brackets(self, entry_price: float, atr: float, side: int,
                      res7: float = np.nan, sup7: float = np.nan):
        if atr is None or np.isnan(atr) or atr <= 0:
            self.cur_stop, self.cur_take = None, None
            return

        # 结构位缓冲：避免贴着压力/支撑放止损
        struct_buf = 0.5 * atr

        if side == 1:
            # 1) 先算“距离止损/止盈”
            stop_dist = entry_price - self.stop_atr * atr
            take = entry_price + self.take_atr * atr

            # 2) 再算“结构止损”（有 sup7 才启用）
            if sup7 is not None and np.isfinite(sup7):
                stop_struct = sup7 - struct_buf
                # 多单止损：取更低的那个（更远、更不容易被洗）
                stop = min(stop_dist, stop_struct)
            else:
                stop = stop_dist

            self.cur_stop = float(stop)
            self.cur_take = float(take)

        else:
            stop_dist = entry_price + self.stop_atr * atr
            take = entry_price - self.take_atr * atr

            if res7 is not None and np.isfinite(res7):
                stop_struct = res7 + struct_buf
                # 空单止损：取更高的那个（更远、更不容易被洗）
                stop = max(stop_dist, stop_struct)
            else:
                stop = stop_dist

            self.cur_stop = float(stop)
            self.cur_take = float(take)

    def _update_trailing_stop(self, close: float, atr: float, side: int,
                              res7: float = np.nan, sup7: float = np.nan):

        if not self.use_trailing:
            return
        if self.cur_stop is None:
            return
        if self.entry_price is None:
            return
        if atr is None or np.isnan(atr) or atr <= 0:
            return

        # ===== 盈利阈值判断（核心改造）=====
        profit = (close - self.entry_price) if side == 1 else (self.entry_price - close)

        # 未达到启动追踪止损条件 -> 不更新 stop
        if profit < self.trail_start_atr * atr:
            return


        # ===== 开始追踪 =====
        struct_buf = 0.5 * atr  # 结构缓冲（可参数化）

        if side == 1:
            # 多头追踪
            new_stop = close - self.stop_atr * atr

            # 不允许止损推回支撑位之上（避免回踩扫）
            if np.isfinite(sup7):
                new_stop = min(new_stop, sup7 - struct_buf)

            # 只允许向盈利方向移动
            self.cur_stop = max(self.cur_stop, new_stop)

        else:
            # 空头追踪
            new_stop = close + self.stop_atr * atr

            # 不允许止损推回压力位之下
            if np.isfinite(res7):
                new_stop = max(new_stop, res7 + struct_buf)

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
        self.entry_price = None

        return {"exit_reason": reason, "exit_price": float(fill)}

    def run(self, df_signals: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for ts, row in df_signals.iterrows():
            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            volatility = float(row["volatility"])
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
                    hit_stop = low <= self.cur_stop # 止损线
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

                    st = self.portfolio.state
                    if st.position != 0:
                        self.entry_price = float(fill)  # ✅ 记录入场价（用于 trailing 启用判断）

                        side = 1 if st.position > 0 else -1
                        res7 = float(row.get("resistance_7d", np.nan))
                        sup7 = float(row.get("support_7d", np.nan))
                        self._set_brackets(entry_price=float(fill), atr=atr, side=side, res7=res7, sup7=sup7)
                    else:
                        self.cur_stop, self.cur_take = None, None
                        self.entry_price = None  # ✅

            # ========== C) 持仓期间追踪止损 ==========
            trailing_active = False
            st = self.portfolio.state
            if st.position != 0 and self.entry_price is not None and pd.notna(atr) and atr > 0:
                side = 1 if st.position > 0 else -1
                profit = (close - self.entry_price) if side == 1 else (self.entry_price - close)
                trailing_active = profit >= self.trail_start_atr * atr

                if trailing_active:
                    res7 = float(row.get("resistance_7d", np.nan))
                    sup7 = float(row.get("support_7d", np.nan))
                    self._update_trailing_stop(close=close, atr=atr, side=side, res7=res7, sup7=sup7)

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
                # "volatility": volatility,

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
                "trailing_active": trailing_active,
            })

        out = pd.DataFrame(rows).set_index("time")
        out["returns"] = out["equity"].pct_change().fillna(0)
        out["cum_returns"] = (1 + out["returns"]).cumprod()
        return out
