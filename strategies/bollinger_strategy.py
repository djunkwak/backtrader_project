import backtrader as bt

class BollingerStrategy(bt.Strategy):
    params = (
        ('period', 20),  # 볼린저 밴드 기간
        ('devfactor', 2)  # 표준편차 계수
    )

    def __init__(self):
        self.bband = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.params.period, 
            devfactor=self.params.devfactor
        )

    def next(self):
        if not self.position:  # 포지션이 없을 때
            if self.data.close[0] <= self.bband.lines.bot[0]:  # 하단 밴드 터치/돌파
                self.buy()
        else:  # 포지션이 있을 때
            if self.data.close[0] >= self.bband.lines.top[0]:  # 상단 밴드 터치/돌파
                self.close() 