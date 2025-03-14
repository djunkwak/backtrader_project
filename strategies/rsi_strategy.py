import backtrader as bt

class RSIStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),  # RSI 기간
        ('rsi_overbought', 70),  # 과매수 기준
        ('rsi_oversold', 30)  # 과매도 기준
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(
            self.data.close, period=self.params.rsi_period)
        self.trades = []  # 매매 기록

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.trades.append({
                    'date': self.datas[0].datetime.date(),
                    'type': 'buy',
                    'price': order.executed.price
                })
            else:
                self.trades.append({
                    'date': self.datas[0].datetime.date(),
                    'type': 'sell',
                    'price': order.executed.price
                })

    def next(self):
        if not self.position:  # 포지션이 없을 때
            if self.rsi < self.params.rsi_oversold:  # 과매도 상태
                self.buy()
        else:  # 포지션이 있을 때
            if self.rsi > self.params.rsi_overbought:  # 과매수 상태
                self.close() 