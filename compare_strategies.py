import backtrader as bt
import datetime
import yfinance as yf
import pandas as pd
from strategies.sma_cross import SmaCross
from strategies.rsi_strategy import RSIStrategy
from strategies.bollinger_strategy import BollingerStrategy

def run_strategy(strategy_class, data, initial_cash=100000.0):
    # Cerebro 객체 생성
    cerebro = bt.Cerebro()
    
    # 데이터 추가
    cerebro.adddata(data)
    
    # 전략 추가
    cerebro.addstrategy(strategy_class)
    
    # 초기 자본 설정
    cerebro.broker.setcash(initial_cash)
    
    # 수수료 설정
    cerebro.broker.setcommission(commission=0.001)
    
    # 초기 포트폴리오 가치
    start_value = cerebro.broker.getvalue()
    
    # 백테스팅 실행
    cerebro.run()
    
    # 최종 포트폴리오 가치
    final_value = cerebro.broker.getvalue()
    
    # 수익률 계산
    returns = (final_value - start_value) / start_value * 100
    
    return {
        'Strategy': strategy_class.__name__,
        'Final Value': final_value,
        'Return (%)': returns
    }

def main():
    # 데이터 다운로드
    ticker = "TSLA"
    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime.now()
    
    print(f"데이터 다운로드 중... ({ticker})")
    
    try:
        df = yf.download(ticker, start=start, end=end)
        
        if df.empty:
            print("데이터를 가져올 수 없습니다. 다시 시도해주세요.")
            return
            
        print("데이터 다운로드 완료")
        print(f"데이터 기간: {df.index[0]} ~ {df.index[-1]}")
        
        # DataFrame 구조 확인 및 전처리
        print("\nDataFrame 컬럼:", df.columns)
        
        # 멀티인덱스 처리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Backtrader 피드로 변환
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # 인덱스를 날짜로 사용
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1
        )
        
        # 전략 리스트
        strategies = [SmaCross, RSIStrategy, BollingerStrategy]
        
        # 각 전략 실행 및 결과 저장
        results = []
        for strategy in strategies:
            print(f"\n{strategy.__name__} 전략 테스트 중...")
            result = run_strategy(strategy, data)
            results.append(result)
            print(f"최종 포트폴리오 가치: ${result['Final Value']:.2f}")
            print(f"수익률: {result['Return (%)']:.2f}%")
        
        # 결과 비교
        print("\n=== 전략 비교 결과 ===")
        results_df = pd.DataFrame(results)
        results_df = results_df.set_index('Strategy')
        print(results_df)
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        print("프로그램을 다시 실행해주세요.")

if __name__ == "__main__":
    main() 