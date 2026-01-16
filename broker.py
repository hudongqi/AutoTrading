class SimBroker:

    def __init__(self, fee_rate=0.0, slippage_bps=0.0):
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps

    def fill_price(self, mid_price, qty):
        slip = mid_price * self.slippage_bps / 10000
        return mid_price + slip if qty > 0 else mid_price - slip

    def commission(self, notional):
        return abs(notional) * self.fee_rate
