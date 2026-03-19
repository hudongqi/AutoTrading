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
        take_R: float = 3.5,
        trail_start_R: float = 1.5,
        trail_atr: float = 2.0,
        use_trailing: bool = True,
        check_liq: bool = True,  # 是否做简化爆仓检查
        entry_is_maker: bool = False,
        funding_rate_per_8h: float = 0.0,
        risk_per_trade: float = 0.0075,
        enable_risk_position_sizing: bool = True,
        allow_reentry: bool = True,
        partial_take_R: float = 0.0,
        partial_take_frac: float = 0.0,
        break_even_after_partial: bool = False,
        break_even_R: float = 0.0,
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
        self.entry_is_maker = bool(entry_is_maker)
        self.funding_rate_per_8h = float(funding_rate_per_8h)
        self.risk_per_trade = float(risk_per_trade)
        self.enable_risk_position_sizing = bool(enable_risk_position_sizing)
        self.allow_reentry = bool(allow_reentry)
        self.partial_take_R = float(partial_take_R)
        self.partial_take_frac = float(partial_take_frac)
        self.break_even_after_partial = bool(break_even_after_partial)
        self.break_even_R = float(break_even_R)

        self.cur_stop = None
        self.cur_take = None
        self.entry_price = None
        self.entry_risk = None
        self.partial_taken = False

        self.trade_count = 0
        self.reversal_count = 0
        self.closed_trade_pnls = []
        self.closed_trades = []
        self.current_trade = None
        self.rejected_entries = []


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

    def _estimate_entry_stop(self, entry_price: float, atr: float, side: int, res7: float, sup7: float) -> float:
        if atr is None or np.isnan(atr) or atr <= 0:
            return np.nan
        struct_buf = 0.25 * atr
        if side == 1:
            stop_atr = entry_price - self.stop_atr * atr
            stop_struct = (sup7 - struct_buf) if np.isfinite(sup7) else -np.inf
            stop = max(stop_atr, stop_struct)
        else:
            stop_atr = entry_price + self.stop_atr * atr
            stop_struct = (res7 + struct_buf) if np.isfinite(res7) else np.inf
            stop = min(stop_atr, stop_struct)
        return float(stop)

    def _risk_based_target(self, side: int, price: float, atr: float, res7: float, sup7: float) -> float:
        # 返回目标仓位（带方向）
        margin_cap = min(self.max_pos, self.portfolio.max_qty_by_margin(price))
        if margin_cap <= 0:
            return 0.0
        if not self.enable_risk_position_sizing:
            return margin_cap * side

        eq = self.portfolio.equity(price)
        if eq <= 0:
            return 0.0

        est_stop = self._estimate_entry_stop(price, atr, side, res7, sup7)
        if not np.isfinite(est_stop):
            return 0.0
        stop_dist = abs(price - est_stop)
        if stop_dist <= 0:
            return 0.0

        risk_budget = eq * self.risk_per_trade
        qty_by_risk = risk_budget / stop_dist
        qty = min(margin_cap, qty_by_risk)
        return qty * side

    def _update_trade_excursions(self, high: float, low: float):
        if not self.current_trade:
            return
        entry = self.current_trade["entry_price"]
        side = self.current_trade["side"]
        if side == 1:
            mfe = high - entry
            mae = entry - low
        else:
            mfe = entry - low
            mae = high - entry
        self.current_trade["mfe"] = max(self.current_trade.get("mfe", 0.0), float(mfe))
        self.current_trade["mae"] = max(self.current_trade.get("mae", 0.0), float(mae))

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

    def _partial_take(self, exit_mid_price: float, frac: float, reason: str) -> dict:
        st = self.portfolio.state
        if st.position == 0 or frac <= 0 or frac >= 1:
            return {"exit_reason": None, "exit_price": np.nan, "realized": 0.0}

        qty_to_close = -st.position * frac
        fill = self.broker.fill_price(exit_mid_price, qty_to_close)
        fill_info = self.portfolio.apply_fill(fill_price=fill, qty=qty_to_close, is_maker=False)
        self.trade_count += 1

        if self.current_trade is not None:
            self.current_trade["fee_accum"] = self.current_trade.get("fee_accum", 0.0) + float(fill_info.get("fee", 0.0))
            self.current_trade.setdefault("partial_exits", []).append({
                "reason": reason,
                "price": float(fill),
                "qty": float(qty_to_close),
                "realized_gross": float(fill_info.get("realized", 0.0)),
            })

        return {"exit_reason": reason, "exit_price": float(fill), "realized": float(fill_info.get("realized", 0.0))}

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
        realized = float(fill_info.get("realized", 0.0)) if fill_info else 0.0

        # 记录交易明细（MFE/MAE/持仓时长）
        if self.current_trade is not None:
            tr = dict(self.current_trade)
            tr.update({
                "exit_reason": reason,
                "exit_price": float(fill),
                "realized_gross": realized,
            })
            tr_fee = float(tr.get("fee_accum", 0.0))
            tr_funding = float(tr.get("funding_accum", 0.0))
            tr["realized_net"] = realized - tr_fee + tr_funding
            self.closed_trade_pnls.append(float(tr["realized_net"]))
            self.closed_trades.append(tr)
            self.current_trade = None

        # 平仓后清空风控线
        self.cur_stop, self.cur_take = None, None
        self.entry_price = None
        self.entry_risk = None
        self.partial_taken = False

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

            # 记录持仓过程指标
            if pos != 0 and self.current_trade is not None:
                self.current_trade["holding_bars"] = self.current_trade.get("holding_bars", 0) + 1
                self._update_trade_excursions(high=high, low=low)

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
                    partial_take_price = self.entry_price + self.partial_take_R * self.entry_risk if (self.entry_price is not None and self.entry_risk is not None and self.partial_take_R > 0) else None
                    hit_partial = (not self.partial_taken) and (partial_take_price is not None) and (high >= partial_take_price)

                    # 同根同时触发：保守先止损
                    if hit_stop:
                        evt = self._close_position(exit_mid_price=self.cur_stop, reason="STOP")
                        exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                        if exit_reason is not None:
                            last_exit_idx = i
                    elif hit_partial and self.partial_take_frac > 0:
                        evt = self._partial_take(exit_mid_price=partial_take_price, frac=self.partial_take_frac, reason="PARTIAL_TAKE")
                        self.partial_taken = True
                        if self.break_even_after_partial:
                            self.cur_stop = max(self.cur_stop, self.entry_price)
                    elif hit_take:
                        evt = self._close_position(exit_mid_price=self.cur_take, reason="TAKE")
                        exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                        if exit_reason is not None:
                            last_exit_idx = i

                else:
                    hit_stop = high >= self.cur_stop
                    hit_take = low <= self.cur_take
                    partial_take_price = self.entry_price - self.partial_take_R * self.entry_risk if (self.entry_price is not None and self.entry_risk is not None and self.partial_take_R > 0) else None
                    hit_partial = (not self.partial_taken) and (partial_take_price is not None) and (low <= partial_take_price)

                    if hit_stop:
                        evt = self._close_position(exit_mid_price=self.cur_stop, reason="STOP")
                        exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                        if exit_reason is not None:
                            last_exit_idx = i
                    elif hit_partial and self.partial_take_frac > 0:
                        evt = self._partial_take(exit_mid_price=partial_take_price, frac=self.partial_take_frac, reason="PARTIAL_TAKE")
                        self.partial_taken = True
                        if self.break_even_after_partial:
                            self.cur_stop = min(self.cur_stop, self.entry_price)
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

            # ========== B) 再处理交易信号（支持二次入场） ==========
            trade_sig = float(row.get("trade_signal", 0.0))
            signal = int(row.get("signal", 0))
            entry_setup = int(row.get("entry_setup", 0))
            in_cooldown = (i - last_exit_idx) < self.cooldown_bars

            # 若策略提供 entry_setup，用它作为入场触发；否则退回 trade_signal
            entry_trigger = entry_setup if "entry_setup" in row else (signal if trade_sig != 0 else 0)

            st = self.portfolio.state
            current_pos = float(st.position)

            # 趋势反向先平仓
            if signal == 1 and current_pos < 0:
                self.reversal_count += 1
                bar_reversal = True
                evt = self._close_position(exit_mid_price=close, reason="SIG_CLOSE_SHORT")
                if evt["exit_reason"] is not None:
                    exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                    last_exit_idx = i
                current_pos = float(self.portfolio.state.position)

            elif signal == -1 and current_pos > 0:
                self.reversal_count += 1
                bar_reversal = True
                evt = self._close_position(exit_mid_price=close, reason="SIG_CLOSE_LONG")
                if evt["exit_reason"] is not None:
                    exit_reason, exit_price = evt["exit_reason"], evt["exit_price"]
                    last_exit_idx = i
                current_pos = float(self.portfolio.state.position)

            can_reenter = self.allow_reentry or trade_sig != 0
            if current_pos == 0 and (not in_cooldown) and can_reenter and entry_trigger != 0:
                side = 1 if entry_trigger > 0 else -1
                res7 = float(row.get("resistance_7d", np.nan))
                sup7 = float(row.get("support_7d", np.nan))
                target = self._risk_based_target(side=side, price=close, atr=atr, res7=res7, sup7=sup7)
                order_qty = float(self.portfolio.target_position(target, close))

                if order_qty != 0:
                    fill = self.broker.fill_price(close, order_qty)
                    fill_info = self.portfolio.apply_fill(fill_price=fill, qty=order_qty, is_maker=self.entry_is_maker)
                    self.trade_count += 1
                    bar_order_qty = float(order_qty)
                    bar_fee += float(fill_info.get("fee", 0.0)) if fill_info else 0.0

                    st = self.portfolio.state
                    if st.position != 0:
                        self.entry_price = float(fill)
                        self._set_brackets(entry_price=float(fill), atr=atr, side=side, res7=res7, sup7=sup7)
                        self.entry_risk = abs(self.entry_price - self.cur_stop) if self.cur_stop is not None else None
                        self.partial_taken = False
                        self.current_trade = {
                            "entry_time": ts,
                            "entry_index": i,
                            "entry_price": self.entry_price,
                            "side": side,
                            "holding_bars": 0,
                            "mfe": 0.0,
                            "mae": 0.0,
                            "fee_accum": float(fill_info.get("fee", 0.0)) if fill_info else 0.0,
                            "funding_accum": 0.0,
                            "entry_atr": float(atr) if pd.notna(atr) else np.nan,
                            "entry_adx_4h": float(row.get("adx_4h", np.nan)),
                            "entry_regime_ok": bool(row.get("regime_ok", True)),
                            "entry_pullback_depth": float(row.get("pullback_depth_long", np.nan) if side == 1 else row.get("pullback_depth_short", np.nan)),
                            "entry_breakout_quality": bool(row.get("breakout_quality_long", False) if side == 1 else row.get("breakout_quality_short", False)),
                            "breakout_bar_time": row.get("breakout_bar_time_long", None) if side == 1 else row.get("breakout_bar_time_short", None),
                            "breakout_level": float(row.get("breakout_level_long", np.nan) if side == 1 else row.get("breakout_level_short", np.nan)),
                            "breakout_age_at_entry": float(row.get("bars_since_breakout_long", np.nan) if side == 1 else row.get("bars_since_breakout_short", np.nan)),
                            "rejection_type": row.get("rejection_type_long", "none") if side == 1 else row.get("rejection_type_short", "none"),
                            "first_pullback_flag": bool(row.get("first_pullback_ok_long", False) if side == 1 else row.get("first_pullback_ok_short", False)),
                            "entry_reason": "breakout->pullback->rejection",
                            "entry_regime_snapshot": {
                                "regime_ok": bool(row.get("regime_ok", True)),
                                "adx_4h": float(row.get("adx_4h", np.nan)),
                                "trend_strength_4h": float(row.get("trend_strength_4h", np.nan)),
                            },
                            "entry_risk_per_unit": float(self.entry_risk) if self.entry_risk is not None else np.nan,
                            "initial_risk_cash": abs(float(order_qty)) * float(self.entry_risk) if (self.entry_risk is not None and order_qty is not None) else np.nan,
                        }
            elif current_pos == 0 and (not in_cooldown) and signal != 0 and entry_trigger == 0:
                reject_reason = []
                if not bool(row.get("regime_ok", True)):
                    reject_reason.append("regime_not_ok")
                if signal == 1:
                    if not bool(row.get("breakout_quality_long", False)):
                        reject_reason.append("breakout_quality_long_fail")
                    if not bool(row.get("pullback_depth_ok_long", True)):
                        reject_reason.append("pullback_too_deep_long")
                    if not bool(row.get("reject_long", False)):
                        reject_reason.append("rejection_long_fail")
                    if not bool(row.get("first_pullback_ok_long", True)):
                        reject_reason.append("not_first_pullback_long")
                elif signal == -1:
                    if not bool(row.get("breakout_quality_short", False)):
                        reject_reason.append("breakout_quality_short_fail")
                    if not bool(row.get("pullback_depth_ok_short", True)):
                        reject_reason.append("pullback_too_deep_short")
                    if not bool(row.get("reject_short", False)):
                        reject_reason.append("rejection_short_fail")
                    if not bool(row.get("first_pullback_ok_short", True)):
                        reject_reason.append("not_first_pullback_short")
                self.rejected_entries.append({
                    "time": ts,
                    "signal": signal,
                    "adx_4h": float(row.get("adx_4h", np.nan)),
                    "reasons": reject_reason,
                    "pullback_depth_long": float(row.get("pullback_depth_long", np.nan)),
                    "pullback_depth_short": float(row.get("pullback_depth_short", np.nan)),
                    "breakout_age_long": float(row.get("bars_since_breakout_long", np.nan)),
                    "breakout_age_short": float(row.get("bars_since_breakout_short", np.nan)),
                    "breakout_level_long": float(row.get("breakout_level_long", np.nan)),
                    "breakout_level_short": float(row.get("breakout_level_short", np.nan)),
                })

            # ========== B2) Funding 成本/收益模拟 ==========
            funding_cashflow = 0.0
            if self.funding_rate_per_8h != 0.0:
                funding_rate_bar = self.funding_rate_per_8h / 8.0  # 1h bar 折算
                funding_cashflow = self.portfolio.apply_funding(mark_price=close, funding_rate=funding_rate_bar)
                if self.current_trade is not None:
                    self.current_trade["funding_accum"] = self.current_trade.get("funding_accum", 0.0) + float(funding_cashflow)

            # ========== C) 持仓期间追踪止损 ==========
            trailing_active = False
            st = self.portfolio.state
            if st.position != 0 and self.entry_price is not None and pd.notna(atr) and atr > 0:
                side = 1 if st.position > 0 else -1
                profit = (close - self.entry_price) if side == 1 else (self.entry_price - close)

                # break-even 保护
                if self.entry_risk is not None and self.entry_risk > 0 and self.break_even_R > 0:
                    if profit >= self.break_even_R * self.entry_risk:
                        if side == 1:
                            self.cur_stop = max(self.cur_stop, self.entry_price)
                        else:
                            self.cur_stop = min(self.cur_stop, self.entry_price)
                        if self.current_trade is not None:
                            self.current_trade["break_even_armed"] = True

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
                "funding_total": st.funding_total,
                "funding_cashflow": funding_cashflow,
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

        expectancy = float(np.mean(self.closed_trade_pnls)) if self.closed_trade_pnls else 0.0
        gross_profit = float(np.sum(wins)) if wins else 0.0
        gross_loss = abs(float(np.sum(losses))) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
        avg_holding_bars = float(np.mean([t.get("holding_bars", 0) for t in self.closed_trades])) if self.closed_trades else 0.0
        time_in_market = float((out["position"].abs() > 0).mean()) if len(out) else 0.0

        long_trades = [t for t in self.closed_trades if t.get("side") == 1]
        short_trades = [t for t in self.closed_trades if t.get("side") == -1]

        def side_stats(trades):
            if not trades:
                return {
                    "count": 0, "win_rate": 0.0, "profit_factor": np.nan, "expectancy": 0.0,
                    "avg_pnl": 0.0, "avg_mfe": 0.0, "avg_mae": 0.0, "avg_holding_bars": 0.0,
                }
            pnls = [float(t.get("realized_net", 0.0)) for t in trades]
            w = [x for x in pnls if x > 0]; l = [x for x in pnls if x < 0]
            pf = (sum(w) / abs(sum(l))) if l else np.nan
            return {
                "count": len(trades),
                "win_rate": len(w) / len(trades),
                "profit_factor": pf,
                "expectancy": float(np.mean(pnls)),
                "avg_pnl": float(np.mean(pnls)),
                "avg_mfe": float(np.mean([t.get("mfe", 0.0) for t in trades])),
                "avg_mae": float(np.mean([t.get("mae", 0.0) for t in trades])),
                "avg_holding_bars": float(np.mean([t.get("holding_bars", 0) for t in trades])),
            }

        long_side = side_stats(long_trades)
        short_side = side_stats(short_trades)

        mfe_avg = float(np.mean([t.get("mfe", 0.0) for t in self.closed_trades])) if self.closed_trades else 0.0
        mae_avg = float(np.mean([t.get("mae", 0.0) for t in self.closed_trades])) if self.closed_trades else 0.0

        total_net_closed = float(np.sum([t.get("realized_net", 0.0) for t in self.closed_trades])) if self.closed_trades else 0.0
        gross_closed = float(np.sum([t.get("realized_gross", 0.0) for t in self.closed_trades])) if self.closed_trades else 0.0
        fees_per_trade = float(self.portfolio.state.fee_paid) / max(1, len(self.closed_trades))

        # 新指标：MFE capture / give-back / avg R realized / exit reason split
        mfe_capture_vals, giveback_vals, r_realized_vals = [], [], []
        exit_reason_split = {}
        partial_remainder_vals = []
        for t in self.closed_trades:
            mfe = float(t.get("mfe", 0.0))
            net = float(t.get("realized_net", 0.0))
            if mfe > 0:
                mfe_capture_vals.append(net / mfe)
                giveback_vals.append((mfe - net) / mfe)
            init_risk_cash = float(t.get("initial_risk_cash", np.nan))
            if np.isfinite(init_risk_cash) and init_risk_cash > 0:
                r_realized_vals.append(net / init_risk_cash)
            reason = str(t.get("exit_reason", "UNKNOWN"))
            exit_reason_split[reason] = exit_reason_split.get(reason, 0) + 1
            px = t.get("partial_exits", []) or []
            if px:
                rem = net - sum(float(x.get("realized_gross", 0.0)) for x in px)
                partial_remainder_vals.append(rem)

        # 最差20%交易分析
        worst = []
        worst_summary = {}
        if self.closed_trades:
            sorted_tr = sorted(self.closed_trades, key=lambda x: x.get("realized_net", 0.0))
            n = max(1, int(len(sorted_tr) * 0.2))
            worst = sorted_tr[:n]
            worst_summary = {
                "count": n,
                "avg_net": float(np.mean([t.get("realized_net", 0.0) for t in worst])),
                "avg_holding_bars": float(np.mean([t.get("holding_bars", 0) for t in worst])),
                "avg_entry_adx": float(np.mean([t.get("entry_adx_4h", np.nan) for t in worst])),
                "avg_entry_pullback_depth": float(np.mean([t.get("entry_pullback_depth", np.nan) for t in worst])),
                "low_mfe_ratio": float(np.mean([(1 if t.get("mfe", 0.0) < abs(t.get("mae", 0.0)) else 0) for t in worst])),
            }

        reject_reason_count = {}
        for x in self.rejected_entries:
            for r in x.get("reasons", []):
                reject_reason_count[r] = reject_reason_count.get(r, 0) + 1

        out.attrs["stats"] = {
            "trade_count": int(self.trade_count),
            "closed_trade_count": len(self.closed_trades),
            "total_fees": float(self.portfolio.state.fee_paid),
            "funding_total": float(self.portfolio.state.funding_total),
            "reversal_count": int(self.reversal_count),
            "win_rate": float(win_rate),
            "pnl_ratio": float(pnl_ratio) if pd.notna(pnl_ratio) else np.nan,
            "expectancy_per_trade": expectancy,
            "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
            "avg_holding_bars": avg_holding_bars,
            "time_in_market": time_in_market,
            "long_short_split": {"long_count": len(long_trades), "short_count": len(short_trades)},
            "long_side": long_side,
            "short_side": short_side,
            "mfe_avg": mfe_avg,
            "mae_avg": mae_avg,
            "gross_closed_pnl": gross_closed,
            "net_closed_pnl": total_net_closed,
            "fees_per_trade": fees_per_trade,
            "mfe_capture_ratio": float(np.mean(mfe_capture_vals)) if mfe_capture_vals else np.nan,
            "give_back_ratio": float(np.mean(giveback_vals)) if giveback_vals else np.nan,
            "avg_R_realized": float(np.mean(r_realized_vals)) if r_realized_vals else np.nan,
            "exit_reason_split": exit_reason_split,
            "partial_take_effectiveness": float(np.mean(partial_remainder_vals)) if partial_remainder_vals else np.nan,
            "rejected_entries_count": len(self.rejected_entries),
            "rejected_reason_count": reject_reason_count,
            "worst_trades_summary": worst_summary,
        }
        out.attrs["rejected_entries"] = self.rejected_entries
        out.attrs["closed_trades"] = self.closed_trades

        return out
