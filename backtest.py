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
        broker,             # SimBroker（仅滑点）
        portfolio,          # PerpPortfolio
        strategy,           # strategy instance（A版先只透传，不改信号逻辑）
        max_pos: float = 0.1,
        cooldown_bars: int = 3,
        stop_atr: float = 1.5,
        take_R: float = 2.5,
        trail_start_R: float = 1.2,
        trail_atr: float = 2.0,
        use_trailing: bool = True,
        check_liq: bool = True,  # 是否做简化爆仓检查
    ):
        self.broker = broker
        self.portfolio = portfolio
        self.strategy = strategy
        self.max_pos = float(max_pos)
        self.cooldown_bars = int(cooldown_bars)
        self.stop_atr = float(stop_atr)
        self.take_R = float(take_R)
        self.trail_start_R = float(trail_start_R)
        self.trail_atr = float(trail_atr)
        self.use_trailing = bool(use_trailing)
        self.check_liq = bool(check_liq)

        self.cur_stop = None
        self.cur_take = None
        self.entry_price = None
        self.entry_risk = None

        self.trade_count = 0
        self.reversal_count = 0
        self.closed_trade_pnls = []


    def _set_brackets(self, entry_price: float, atr: float, side: int,
                      res7: float = np.nan, sup7: float = np.nan):

        if atr is None or np.isnan(atr) or atr <= 0:
            self.cur_stop, self.cur_take = None, None
            return

        struct_buf = 0.25 * atr  # 缓冲缩小一点（更贴近结构 = 更高R）

        if side == 1:
            stop_atr = entry_price - self.stop_atr * atr
            stop_struct = (sup7 - struct_buf) if np.isfinite(sup7) else -np.inf

            # ✅ 高R：取“更近”的止损（价格更高的那个）
            stop = max(stop_atr, stop_struct)

            risk = entry_price - stop
            if risk <= 0:
                self.cur_stop, self.cur_take = None, None
                return

            take = entry_price + self.take_R * risk

        else:
            stop_atr = entry_price + self.stop_atr * atr
            stop_struct = (res7 + struct_buf) if np.isfinite(res7) else np.inf

            # ✅ 高R：取“更近”的止损（价格更低的那个）
            stop = min(stop_atr, stop_struct)

            risk = stop - entry_price
            if risk <= 0:
                self.cur_stop, self.cur_take = None, None
                return

            take = entry_price - self.take_R * risk

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
        if self.entry_risk is None or self.entry_risk <= 0:
            return

        # ===== 盈利阈值判断（核心改造）=====
        profit = (close - self.entry_price) if side == 1 else (self.entry_price - close)
        if profit < self.trail_start_R * self.entry_risk:
            return

        # ===== 开始追踪 =====
        struct_buf = 0.5 * atr  # 结构缓冲（可参数化）

        if side == 1:
            # 多头追踪
            new_stop = close - self.trail_atr * atr

            # 不允许止损推回支撑位之上（避免回踩扫）
            if np.isfinite(sup7):
                new_stop = min(new_stop, sup7 - struct_buf)

            # 只允许向盈利方向移动
            self.cur_stop = max(self.cur_stop, new_stop)

        else:
            # 空头追踪
            new_stop = close + self.trail_atr * atr

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
        fill_info = self.portfolio.apply_fill(fill_price=fill, qty=qty_to_close, is_maker=False)
        self.trade_count += 1
        if fill_info and fill_info.get("realized", 0.0) != 0:
            self.closed_trade_pnls.append(float(fill_info["realized"]))

        # 平仓后清空风控线
        self.cur_stop, self.cur_take = None, None
        self.entry_price = None
        self.entry_risk = None

        return {"exit_reason": reason, "exit_price": float(fill)}

    def run(self, df_signals: pd.DataFrame) -> pd.DataFrame:
        rows = []
        last_exit_idx = -999999

        for i, (ts, row) in enumerate(df_signals.iterrows()):
            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            volatility = float(row["volatility"])
            atr = float(row["atr"]) if ("atr" in row and pd.notna(row["atr"])) else np.nan

            st = self.portfolio.state
            pos = float(st.position)

            exit_reason = None
            exit_price = np.nan
            bar_fee = 0.0
            bar_order_qty = 0.0
            bar_reversal = False

            # ========== (可选) 简化爆仓检查 ==========
            if self.check_liq and self.portfolio.is_liquidation_risk(close):
                evt = self._close_position(exit_mid_price=close, reason="LIQ")
                exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                if exit_reason is not None:
                    last_exit_idx = i
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
                        if exit_reason is not None:
                            last_exit_idx = i
                    elif hit_take:
                        evt = self._close_position(exit_mid_price=self.cur_take, reason="TAKE")
                        exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                        if exit_reason is not None:
                            last_exit_idx = i

                else:
                    hit_stop = high >= self.cur_stop
                    hit_take = low <= self.cur_take

                    if hit_stop:
                        evt = self._close_position(exit_mid_price=self.cur_stop, reason="STOP")
                        exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                        if exit_reason is not None:
                            last_exit_idx = i
                    elif hit_take:
                        evt = self._close_position(exit_mid_price=self.cur_take, reason="TAKE")
                        exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                        if exit_reason is not None:
                            last_exit_idx = i

            # 重新读取持仓（因为可能刚刚止损/止盈/爆仓平了）
            st = self.portfolio.state
            pos = float(st.position)

            # ========== B0) 趋势失效先平仓（但不反手） ==========
            signal = int(row["signal"])
            if pos > 0 and signal == -1:
                self.reversal_count += 1
                bar_reversal = True
                evt = self._close_position(exit_mid_price=close, reason="TREND_EXIT_LONG")
                if evt["exit_reason"] is not None:
                    exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                    last_exit_idx = i

            elif pos < 0 and signal == 1:
                self.reversal_count += 1
                bar_reversal = True
                evt = self._close_position(exit_mid_price=close, reason="TREND_EXIT_SHORT")
                if evt["exit_reason"] is not None:
                    exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                    last_exit_idx = i

            # 重新读取持仓
            st = self.portfolio.state
            pos = float(st.position)

            # ========== B) 再处理交易信号（事件驱动） ==========
            trade_sig = float(row["trade_signal"])
            signal = int(row["signal"])
            in_cooldown = (i - last_exit_idx) < self.cooldown_bars

            if trade_sig != 0:
                st = self.portfolio.state
                current_pos = float(st.position)

                # A版：禁止直接反手
                if signal == 1:
                    if current_pos < 0:
                        self.reversal_count += 1
                        bar_reversal = True
                        evt = self._close_position(exit_mid_price=close, reason="SIG_CLOSE_SHORT")
                        if evt["exit_reason"] is not None:
                            exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                            last_exit_idx = i
                    elif current_pos == 0 and not in_cooldown:
                        target = self.max_pos
                        order_qty = float(self.portfolio.target_position(target, close))
                        if order_qty != 0:
                            fill = self.broker.fill_price(close, order_qty)
                            fill_info = self.portfolio.apply_fill(fill_price=fill, qty=order_qty, is_maker=False)
                            self.trade_count += 1
                            bar_order_qty = float(order_qty)
                            bar_fee += float(fill_info.get("fee", 0.0)) if fill_info else 0.0

                            st = self.portfolio.state
                            if st.position != 0:
                                self.entry_price = float(fill)
                                side = 1
                                res7 = float(row.get("resistance_7d", np.nan))
                                sup7 = float(row.get("support_7d", np.nan))
                                self._set_brackets(entry_price=float(fill), atr=atr, side=side, res7=res7, sup7=sup7)
                                self.entry_risk = abs(self.entry_price - self.cur_stop) if self.cur_stop is not None else None

                elif signal == -1:
                    if current_pos > 0:
                        self.reversal_count += 1
                        bar_reversal = True
                        evt = self._close_position(exit_mid_price=close, reason="SIG_CLOSE_LONG")
                        if evt["exit_reason"] is not None:
                            exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                            last_exit_idx = i
                    elif current_pos == 0 and not in_cooldown:
                        target = -self.max_pos
                        order_qty = float(self.portfolio.target_position(target, close))
                        if order_qty != 0:
                            fill = self.broker.fill_price(close, order_qty)
                            fill_info = self.portfolio.apply_fill(fill_price=fill, qty=order_qty, is_maker=False)
                            self.trade_count += 1
                            bar_order_qty = float(order_qty)
                            bar_fee += float(fill_info.get("fee", 0.0)) if fill_info else 0.0

                            st = self.portfolio.state
                            if st.position != 0:
                                self.entry_price = float(fill)
                                side = -1
                                res7 = float(row.get("resistance_7d", np.nan))
                                sup7 = float(row.get("support_7d", np.nan))
                                self._set_brackets(entry_price=float(fill), atr=atr, side=side, res7=res7, sup7=sup7)
                                self.entry_risk = abs(self.entry_price - self.cur_stop) if self.cur_stop is not None else None

            # ========== C) 持仓期间追踪止损 ==========
            trailing_active = False
            st = self.portfolio.state
            if st.position != 0 and self.entry_price is not None and pd.notna(atr) and atr > 0:
                side = 1 if st.position > 0 else -1
                profit = (close - self.entry_price) if side == 1 else (self.entry_price - close)
                trailing_active = (
                    self.entry_risk is not None and self.entry_risk > 0 and
                    profit >= self.trail_start_R * self.entry_risk
                )

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
                "fee_paid": st.fee_paid,
                "unrealized_pnl": upnl,
                "equity": eq,

                "margin_used": m_used,
                "free_margin": f_margin,

                "stop_price": self.cur_stop,
                "take_price": self.cur_take,

                "signal": signal,
                "trade_signal": trade_sig,
                "order_qty": bar_order_qty,
                "bar_fee": bar_fee,
                "is_reversal": bar_reversal,

                "exit_reason": exit_reason,
                "exit_price": exit_price,
                "liq_risk": self.portfolio.is_liquidation_risk(close),
                "trailing_active": trailing_active,
            })

        out = pd.DataFrame(rows).set_index("time")
        out["returns"] = out["equity"].pct_change().fillna(0)
        out["cum_returns"] = (1 + out["returns"]).cumprod()

        wins = [p for p in self.closed_trade_pnls if p > 0]
        losses = [p for p in self.closed_trade_pnls if p < 0]
        win_rate = (len(wins) / len(self.closed_trade_pnls)) if self.closed_trade_pnls else 0.0
        pnl_ratio = (np.mean(wins) / abs(np.mean(losses))) if wins and losses else np.nan

        out.attrs["stats"] = {
            "trade_count": int(self.trade_count),
            "total_fees": float(self.portfolio.state.fee_paid),
            "reversal_count": int(self.reversal_count),
            "win_rate": float(win_rate),
            "pnl_ratio": float(pnl_ratio) if pd.notna(pnl_ratio) else np.nan,
        }

        return out
