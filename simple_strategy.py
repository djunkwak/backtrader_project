import backtrader as bt
import datetime
import yfinance as yf
import pandas as pd

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

# Cerebro 객체 생성
cerebro = bt.Cerebro()

# yfinance를 사용하여 데이터 가져오기
ticker = "AAPL"
start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2023, 12, 31)

# 데이터 다운로드
df = yf.download(ticker, start=start, end=end)

# DataFrame의 멀티인덱스를 단일 인덱스로 변환
df.columns = [col[0].lower() for col in df.columns]

# DataFrame 구조 확인
print("DataFrame 정보:")
print(df.info())
print("\nDataFrame 컬럼:")
print(df.columns)

# Backtrader 피드로 변환
data = bt.feeds.PandasData(
    dataname=df,
    datetime=None,  # 인덱스를 날짜로 사용
    open='open',
    high='high',
    low='low',
    close='close',
    volume='volume',
    openinterest=-1
)
cerebro.adddata(data)

# 전략 추가
cerebro.addstrategy(SmaCross)

# 초기 자본 설정
cerebro.broker.setcash(100000.0)

# 수수료 설정
cerebro.broker.setcommission(commission=0.001)

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# 백테스팅 실행
cerebro.run()

print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# 결과 플롯
cerebro.plot() 