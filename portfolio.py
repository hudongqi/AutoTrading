from dataclasses import dataclass

@dataclass
class PerpState:
    cash: float                 # 可用现金/余额（USDT）
    position: float = 0.0       # 合约数量（正=多，负=空；这里用“标的数量”，例如 BTC 数量）
    avg_price: float = 0.0      # 持仓均价
    realized_pnl: float = 0.0   # 已实现盈亏累计（USDT）

class PerpPortfolio:
    """
    简化 USDT 永续 Portfolio（单标的/单仓位）：
    - position: 以标的数量计（BTC 数量）。如果你用“张”，可以把 qty 当张数，price 当张面值换算后的价格。
    - equity = cash + unrealized_pnl
    - margin_used = |position| * price / leverage
    - free_margin = equity - margin_used
    """

    def __init__(
        self,
        initial_cash: float,
        leverage: float = 10.0,
        taker_fee_rate: float = 0.0004,   # 例：0.04%
        maker_fee_rate: float = 0.0002,   # 例：0.02%（如果你不区分 maker/taker，可只用 taker）
        maint_margin_rate: float = 0.005  # 例：0.5%，用于简化爆仓检查（可不用）
    ):
        self.state = PerpState(cash=initial_cash)
        self.leverage = float(leverage)
        self.taker_fee_rate = float(taker_fee_rate)
        self.maker_fee_rate = float(maker_fee_rate)
        self.maint_margin_rate = float(maint_margin_rate)

    # ---------- 估值相关 ----------
    def unrealized_pnl(self, mark_price: float) -> float:
        st = self.state
        if st.position == 0:
            return 0.0
        # 多头： (mark - avg) * qty
        # 空头： qty 为负 => 同样公式成立
        return (float(mark_price) - st.avg_price) * st.position

    def equity(self, mark_price: float) -> float:
        # 账户权益 = 现金余额 + 未实现盈亏
        return self.state.cash + self.unrealized_pnl(mark_price)

    def margin_used(self, mark_price: float) -> float:
        # 占用保证金（简化）= 名义价值 / 杠杆
        notional = abs(self.state.position) * float(mark_price)
        return notional / self.leverage

    def free_margin(self, mark_price: float) -> float:
        # 可用保证金 = 权益 - 占用保证金
        return self.equity(mark_price) - self.margin_used(mark_price)

    def is_liquidation_risk(self, mark_price: float) -> bool:
        """
        简化爆仓判断：权益 <= 维持保证金
        维持保证金（简化）= |pos|*price*maint_margin_rate
        """
        st = self.state
        if st.position == 0:
            return False
        maint = abs(st.position) * float(mark_price) * self.maint_margin_rate
        return self.equity(mark_price) <= maint

    # ---------- 下单相关 ----------
    def commission(self, notional: float, is_maker: bool = False) -> float:
        rate = self.maker_fee_rate if is_maker else self.taker_fee_rate
        return abs(notional) * rate

    def max_qty_by_margin(self, price: float) -> float:
        """
        在当前权益下，最多能开到多少“绝对仓位数量”（简化：忽略维持保证金、只看初始保证金）
        max_notional = equity * leverage
        """
        eq = self.equity(price)
        if eq <= 0:
            return 0.0
        max_notional = eq * self.leverage
        return max_notional / float(price)

    def target_position(self, target_qty: float, price: float) -> float:
        """
        返回需要下单的数量（买为+，卖为-）。
        增加一个简易风控：目标仓位不能超过保证金允许的最大仓位。
        """
        price = float(price)
        target_qty = float(target_qty)

        # 限制目标仓位：不超过当前权益允许的最大仓位
        max_abs = self.max_qty_by_margin(price)
        if max_abs > 0:
            if target_qty > max_abs:
                target_qty = max_abs
            elif target_qty < -max_abs:
                target_qty = -max_abs

        return target_qty - self.state.position

    def apply_fill(self, fill_price: float, qty: float, is_maker: bool = False) -> None:
        """
        把一笔成交记到账户里（支持开仓/加仓/减仓/平仓/反手）
        - fill_price: 成交价
        - qty: 成交数量（买+，卖-）
        """
        st = self.state
        fill_price = float(fill_price)
        qty = float(qty)
        if qty == 0:
            return

        # 手续费按成交名义收取（简化：直接从 cash 扣）
        notional = fill_price * qty
        fee = self.commission(notional, is_maker=is_maker)
        st.cash -= fee

        old_pos = st.position
        old_avg = st.avg_price
        new_pos = old_pos + qty

        # 1) 原来无仓位 -> 直接开仓
        if old_pos == 0:
            st.position = new_pos
            st.avg_price = fill_price
            return

        # 2) 同方向加仓（old_pos 与 qty 同号）
        if (old_pos > 0 and qty > 0) or (old_pos < 0 and qty < 0):
            total_cost = old_avg * abs(old_pos) + fill_price * abs(qty)
            st.position = new_pos
            st.avg_price = total_cost / abs(new_pos)
            return

        # 3) 反方向交易：减仓 / 平仓 / 反手
        # 先计算此次成交中“平掉的数量”（绝对值）
        close_qty = min(abs(qty), abs(old_pos))  # 被平掉的数量（正数）
        # 已实现盈亏：多头平仓 = (平仓价 - 均价)*平仓数量；空头也用同一公式（old_pos 为负会自然处理不直观）
        # 为了更直观，用方向来写：
        if old_pos > 0:
            # 多头被卖出平仓：qty < 0
            realized = (fill_price - old_avg) * close_qty
        else:
            # 空头被买入平仓：qty > 0
            realized = (old_avg - fill_price) * close_qty

        st.realized_pnl += realized
        st.cash += realized  # 简化：已实现盈亏直接回到现金

        # 更新仓位
        st.position = new_pos

        # 3.1) 如果刚好平完（新仓位=0）
        if new_pos == 0:
            st.avg_price = 0.0
            return

        # 3.2) 如果反手（新仓位方向与旧仓位相反）
        # 反手后剩余那部分是“新开仓”，均价=本次成交价
        if (old_pos > 0 and new_pos < 0) or (old_pos < 0 and new_pos > 0):
            st.avg_price = fill_price
            return

        # 3.3) 否则是减仓但还在同方向（均价通常保持不变）
        # st.avg_price 保持 old_avg
        st.avg_price = old_avg
