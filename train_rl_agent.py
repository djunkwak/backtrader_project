import yfinance as yf
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from rl_env import PortfolioOptEnv
from strategies.sma_cross import SmaCross
from strategies.rsi_strategy import RSIStrategy
from strategies.bollinger_strategy import BollingerStrategy
import matplotlib.font_manager as fm
import backtrader as bt
import os
import torch
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from copy import deepcopy
import torch.nn as nn

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac OS용 한글 폰트
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

# 상수 정의
TICKERS = ['^GSPC']  # S&P 500 지수
TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2023-12-31'
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = '2025-03-14'

# models 디렉토리 생성
if not os.path.exists('models'):
    os.makedirs('models')

def download_and_preprocess_data(symbol, start_date, end_date):
    """데이터 다운로드 및 전처리"""
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"{symbol} 데이터를 찾을 수 없습니다.")
            
        # 필요한 열만 선택
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # 결측값 처리
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        print(f"{symbol} 데이터 다운로드 완료: {len(df)} 행")
        return df
        
    except Exception as e:
        print(f"{symbol} 데이터 다운로드 실패: {str(e)}")
        return None

def train_model(env, total_timesteps=100000):
    """강화학습 모델 학습"""
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # 학습률 증가 (이전 3e-4)
        n_steps=1024,        # 스텝 크기 감소 (이전 2048)
        batch_size=128,      # 배치 크기 감소 (이전 256)
        n_epochs=10,         # 에포크 수 감소 (이전 15)
        gamma=0.95,          # 할인율 감소 (이전 0.97)
        gae_lambda=0.9,
        clip_range=0.25,     # 클리핑 범위 증가 (이전 0.2)
        ent_coef=0.1,        # 엔트로피 계수 크게 증가 (이전 0.01)
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 64],     # 더 단순한 정책 네트워크 (이전 [256, 128, 64])
                vf=[128, 64]      # 더 단순한 가치 네트워크 (이전 [256, 128, 64])
            ),
            activation_fn=nn.ReLU
        ),
        verbose=1
    )
    
    # 콜백 설정
    eval_callback = EvalCallback(
        eval_env=env,
        n_eval_episodes=5,   # 평가 에피소드 감소 (이전 10)
        eval_freq=2000,      # 더 자주 평가 (이전 5000)
        log_path="./logs",
        best_model_save_path="./models/best_model",
        deterministic=True,
        render=False
    )
    
    # 모델 학습
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    return model

def evaluate_model(model, env, n_eval_episodes=1):
    """모델 평가"""
    try:
        # 직접 환경에서 액션 실행
        env_base = env.envs[0].unwrapped
        env_base.reset()
        
        total_reward = 0
        initial_value = env_base.portfolio_value
        all_actions = []
        steps = 0
        
        action_counts = {0: 0, 1: 0, 2: 0}  # 액션 카운트 추가
        
        # 500 스텝 실행
        for _ in range(500):
            if env_base.current_step >= len(env_base.df) - 1:
                break
                
            # 현재 상태 관찰
            obs = env_base._get_observation()
            
            # 모델에서 행동 예측 - 확률적인 선택으로 변경
            action, _ = model.predict([obs], deterministic=False)  # deterministic=False로 변경
            action = action[0]
            all_actions.append(int(action))
            action_counts[int(action)] += 1  # 액션 카운트 증가
            
            # 환경 진행
            _, reward, done, _, info = env_base.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # 결과 계산
        final_value = env_base.portfolio_value
        returns = (final_value - initial_value) / initial_value
        
        # 액션 비율 계산
        total_actions = sum(action_counts.values())
        action_ratios = {k: (v / total_actions * 100) for k, v in action_counts.items()}
        print(f"\n액션 비율: 매도 {action_ratios[0]:.1f}%, 홀드 {action_ratios[1]:.1f}%, 매수 {action_ratios[2]:.1f}%")
        
        return total_reward, returns, steps, all_actions
    
    except Exception as e:
        print(f"평가 중 오류 발생: {str(e)}")
        return 0.0, 0.0, 0, []

def run_traditional_strategy(df, strategy_class):
    """전통적인 전략 실행"""
    cerebro = bt.Cerebro()
    
    # 데이터 준비
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # 인덱스가 날짜
        open=0,         # 'Open' 컬럼
        high=1,         # 'High' 컬럼
        low=2,          # 'Low' 컬럼
        close=3,        # 'Close' 컬럼
        volume=4,       # 'Volume' 컬럼
        openinterest=-1 # 사용하지 않음
    )
    
    cerebro.adddata(data)
    cerebro.addstrategy(strategy_class)
    cerebro.broker.setcash(100000.0)
    
    # 전략 실행
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    # 수익률 계산
    return_pct = (final_value - initial_value) / initial_value * 100
    
    # 매매 기록 디버깅
    trades = results[0].trades
    print(f"\n{strategy_class.__name__} 전략 거래 기록:")
    print(f"총 거래 수: {len(trades)}")
    
    # 매매 기록 반환
    return return_pct, trades

def plot_results(dates, prices, actions, name):
    """거래 결과 시각화"""
    plt.figure(figsize=(15, 7))
    plt.plot(dates, prices, label='주가', alpha=0.7)
    
    # 액션 배열 준비
    actions = np.array(actions)
    if len(actions) < len(dates):
        actions = np.pad(actions, (0, len(dates) - len(actions)), 'constant', constant_values=1)  # 기본값을 홀드(1)로 설정
    elif len(actions) > len(dates):
        actions = actions[:len(dates)]
    
    # 액션 레이블 설정 (이산적 액션)
    buy_signals = np.where(actions == 2)[0]   # 매수 (action = 2)
    sell_signals = np.where(actions == 0)[0]  # 매도 (action = 0)
    hold_signals = np.where(actions == 1)[0]  # 홀드 (action = 1)
    
    # 매수 시그널
    if len(buy_signals) > 0:
        plt.scatter(dates[buy_signals], prices[buy_signals], 
                   marker='^', color='g', label='매수', alpha=1)
    
    # 매도 시그널
    if len(sell_signals) > 0:
        plt.scatter(dates[sell_signals], prices[sell_signals], 
                   marker='v', color='r', label='매도', alpha=1)
    
    # 홀드 시그널
    if len(hold_signals) > 0:
        plt.scatter(dates[hold_signals], prices[hold_signals], 
                   marker='o', color='b', label='홀드', alpha=0.5)
    
    # 거래 통계 출력
    total_trades = len(actions)
    buy_count = len(buy_signals)
    sell_count = len(sell_signals)
    hold_count = len(hold_signals)
    
    print(f"\n{name} 거래 통계:")
    print(f"총 데이터 포인트: {total_trades}")
    print(f"매수 횟수: {buy_count} ({buy_count/total_trades*100:.1f}%)")
    print(f"매도 횟수: {sell_count} ({sell_count/total_trades*100:.1f}%)")
    print(f"홀드 횟수: {hold_count} ({hold_count/total_trades*100:.1f}%)")
    
    plt.title(f'{name} 거래 전략 결과')
    plt.xlabel('날짜')
    plt.ylabel('가격')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 로그 디렉토리 생성
    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)
    os.makedirs("models/best_model", exist_ok=True)
    
    print("트레이닝 데이터 다운로드 중...")
    train_data = download_and_preprocess_data('^GSPC', '2015-01-01', '2020-12-31')
    
    print("\n테스트 데이터 다운로드 중...")
    test_data = download_and_preprocess_data('^GSPC', '2021-01-01', '2022-12-31')
    
    if train_data is None or test_data is None:
        print("데이터 다운로드 실패")
        return
        
    print("\n^GSPC 모델 학습 시작...")
    
    try:
        # 학습 환경 생성
        train_env = PortfolioOptEnv(train_data, initial_balance=37037)  # 원화 5000만원에 해당하는 약 37,037 달러
        train_env = Monitor(train_env, "logs/train")
        train_env = DummyVecEnv([lambda: train_env])
        
        # 모델 학습
        model = train_model(train_env, total_timesteps=100000)
        print(f"^GSPC 모델 저장 완료")
        
        # 테스트 환경 생성
        test_env = PortfolioOptEnv(test_data, initial_balance=37037)  # 원화 5000만원에 해당하는 약 37,037 달러
        test_env = Monitor(test_env, "logs/test")
        test_env = DummyVecEnv([lambda: test_env])
        
        # 모델 평가
        print(f"^GSPC 테스트 결과:")
        try:
            total_reward, returns, steps, all_actions = evaluate_model(model, test_env)
            print(f"총 보상: {total_reward:.2f}")
            print(f"수익률: {returns:.2%}")
            print(f"거래 기간: {steps}")
            print(f"주식 보유량: {test_env.envs[0].unwrapped.shares}주")
            print(f"현금 잔고: ${test_env.envs[0].unwrapped.balance:.2f}")
            
            # 결과 시각화
            plot_results(test_data.index[:min(len(test_data), len(all_actions))], 
                        test_data['Close'].values[:min(len(test_data), len(all_actions))], 
                        all_actions, '^GSPC')
        except Exception as eval_error:
            print(f"^GSPC 모델 평가 중 오류 발생: {str(eval_error)}")
            # 기본 거래 통계 출력
            print("\n기본 거래 통계:")
            print("모델이 아직 거래를 수행하지 않음")
        
    except Exception as e:
        print(f"^GSPC 모델 학습 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 