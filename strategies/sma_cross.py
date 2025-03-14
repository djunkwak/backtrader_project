import backtrader as bt

class SmaCross(bt.Strategy):
    params = (
        ('fast_period', 10),  # 단기 이동평균 기간
        ('slow_period', 30),  # 장기 이동평균 기간
    )

    def __init__(self):
        self.fast_sma = bt.indicators.SMA(
            self.data.close, period=self.params.fast_period)
        self.slow_sma = bt.indicators.SMA(
            self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

    def next(self):
        if not self.position:  # 포지션이 없을 때
            if self.crossover > 0:  # 골든 크로스
                self.buy()
        elif self.crossover < 0:  # 데드 크로스
            self.close() 