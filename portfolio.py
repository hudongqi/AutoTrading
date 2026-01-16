from dataclasses import dataclass

@dataclass
class PortfolioState:
    cash: float
    position: float = 0
    avg_price: float = 0.0


class Portfolio:

    def __init__(self, initial_cash):
        self.state = PortfolioState(cash=initial_cash)

    def value(self, price):
        return self.state.cash + self.state.position * price

    def target_position(self, target_qty, price):
        return target_qty - self.state.position
