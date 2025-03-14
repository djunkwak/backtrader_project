import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class PortfolioOptEnv(gym.Env):
    """포트폴리오 최적화를 위한 강화학습 환경"""
    
    def __init__(self, df, initial_balance=37037):
        super(PortfolioOptEnv, self).__init__()
        
        # 데이터 전처리
        self.df = self.add_indicators(df)
        self.initial_balance = initial_balance
        
        # 스케일링 팩터 설정
        self.price_scale = np.max(self.df['Close'])
        self.balance_scale = initial_balance
        self.rsi_scale = 100.0
        self.position_scale = 1.0
        
        # 초기 포트폴리오 값 설정
        self.initial_portfolio_value = initial_balance
        
        # 관찰 공간 정의 (정규화된 값들로 구성)
        obs_low = np.array([
            0,      # 정규화된 가격
            0,      # RSI (0-1)
            0,      # BB 상단
            0,      # BB 하단
            0,      # 정규화된 잔고
            -1,     # 정규화된 포지션 크기 (-1: 최대 매도, 1: 최대 매수)
            0       # 정규화된 포트폴리오 가치
        ])
        obs_high = np.array([
            2,      # 가격이 2배까지 증가 가능
            1,      # RSI (0-1)
            2,      # BB 상단
            2,      # BB 하단
            2,      # 잔고가 2배까지 증가 가능
            1,      # 정규화된 포지션 크기
            2       # 포트폴리오 가치가 2배까지 증가 가능
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # 액션 공간 정의: Discrete(3) - [매도, 홀드, 매수]
        self.action_space = spaces.Discrete(3)
        
        # 환경 초기화
        self.reset()
        
    def add_indicators(self, df):
        """기술적 지표 추가"""
        df = df.copy()
        
        # RSI 계산 (14일)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.finfo(float).eps)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드 계산 (20일)
        sma = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = sma + (2 * std)
        df['BB_lower'] = sma - (2 * std)
        
        # 결측값 처리
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        
        return df
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 20
        self.balance = self.initial_balance * 0.5  # 초기 금액의 50%를 현금으로 (이전 40%)
        
        # 나머지 50%는 주식으로 투자 (이전 60%)
        initial_price = self.df.iloc[self.current_step]['Close']
        initial_price = float(initial_price.iloc[0] if isinstance(initial_price, pd.Series) else initial_price)
        self.shares = int((self.initial_balance * 0.5) / initial_price)
        
        self.portfolio_value = self.balance + self.shares * initial_price
        self.initial_portfolio_value = self.portfolio_value
        self.last_trade_step = 0
        self.last_action = 1  # 초기 액션은 홀드
        self.consecutive_actions = 0  # 연속 행동 카운터
        self.no_trade_steps = 0  # 거래 없이 홀드만 한 연속 스텝 수
        return self._get_observation(), {}
    
    def _get_observation(self):
        """현재 상태의 관찰값 반환"""
        row = self.df.iloc[self.current_step]
        current_price = float(row['Close'].iloc[0] if isinstance(row['Close'], pd.Series) else row['Close'])
        
        # 포지션 크기 계산 (-1 ~ 1)
        position_size = (self.shares * current_price - self.balance) / (self.shares * current_price + self.balance) if (self.shares * current_price + self.balance) > 0 else 0
        
        # Series 값을 스칼라로 변환
        rsi = float(row['RSI'].iloc[0] if isinstance(row['RSI'], pd.Series) else row['RSI'])
        bb_upper = float(row['BB_upper'].iloc[0] if isinstance(row['BB_upper'], pd.Series) else row['BB_upper'])
        bb_lower = float(row['BB_lower'].iloc[0] if isinstance(row['BB_lower'], pd.Series) else row['BB_lower'])
        
        obs = np.array([
            current_price / self.price_scale,
            rsi / self.rsi_scale,
            bb_upper / self.price_scale,
            bb_lower / self.price_scale,
            self.balance / self.balance_scale,
            position_size,
            self.portfolio_value / self.initial_portfolio_value
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        row = self.df.iloc[self.current_step]
        current_price = float(row['Close'].iloc[0] if isinstance(row['Close'], pd.Series) else row['Close'])
        old_portfolio_value = self.portfolio_value
        
        trade_amount = 0
        position_value = self.shares * current_price
        total_value = position_value + self.balance
        current_position_ratio = position_value / total_value if total_value > 0 else 0
        
        # 연속 거래 카운터 업데이트
        if action != 1 and self.last_action == action:  # 같은 액션 반복
            self.consecutive_actions += 1
        else:
            self.consecutive_actions = 0
        
        # 홀드 연속 카운터 업데이트
        if action == 1:  # 홀드
            self.no_trade_steps += 1
        else:
            self.no_trade_steps = 0
        
        # 매수/매도 로직 개선
        if action == 2:  # 매수
            if self.balance > 0 and self.consecutive_actions < 3:  # 연속 매수 제한
                # 포지션 비율 제한 (90%까지는 매수 가능)
                if current_position_ratio < 0.9:  # 이전 0.8
                    buy_amount = min(self.balance, total_value * 0.2)
                    if buy_amount > 50:
                        shares_to_buy = int(buy_amount / current_price)
                        cost = shares_to_buy * current_price
                        if cost <= self.balance:
                            self.shares += shares_to_buy
                            self.balance -= cost
                            trade_amount = cost
                            print(f"매수: {shares_to_buy}주, 가격: {current_price:.2f}, 비용: {cost:.2f}")
        
        elif action == 0:  # 매도
            if self.shares > 0 and self.consecutive_actions < 3:  # 연속 매도 제한
                # 더 낮은 포지션에서도 매도 가능 (10%까지는 매도 가능)
                if current_position_ratio > 0.1:  # 이전 0.2
                    shares_to_sell = int(self.shares * 0.15)
                    if shares_to_sell > 0:
                        gain = shares_to_sell * current_price
                        self.shares -= shares_to_sell
                        self.balance += gain
                        trade_amount = gain
                        print(f"매도: {shares_to_sell}주, 가격: {current_price:.2f}, 이익: {gain:.2f}")
        
        elif action == 1:  # 홀드
            # 홀드는 특별한 처리 없음
            pass
        
        self.portfolio_value = self.balance + self.shares * current_price
        returns = (self.portfolio_value - old_portfolio_value) / old_portfolio_value
        
        rsi = float(row['RSI'].iloc[0] if isinstance(row['RSI'], pd.Series) else row['RSI'])
        bb_upper = float(row['BB_upper'].iloc[0] if isinstance(row['BB_upper'], pd.Series) else row['BB_upper'])
        bb_lower = float(row['BB_lower'].iloc[0] if isinstance(row['BB_lower'], pd.Series) else row['BB_lower'])
        
        # 보상 계산 재조정
        reward = 0
        
        # 1. 포트폴리오 수익률 기반 보상 (가장 중요)
        reward += returns * 10  # 수익률 보상 감소 (이전 20)
        
        # 2. 매수/매도/홀드 신호에 대한 보상 재조정
        if action == 2:  # 매수
            # 과매도 상태나 하단 돌파 시 매수에 큰 보상
            if rsi < 30 or current_price < bb_lower:
                reward += 0.4  # 매수 보상 증가 (이전 0.3)
            # 과매수 상태나 상단 돌파 시 매수에 약한 페널티
            elif rsi > 70 or current_price > bb_upper:
                reward -= 0.03  # 페널티 감소 (이전 0.05)
            # 기본 매수 보상 증가
            else:
                reward += 0.1  # 기본 매수 보상 증가 (이전 0.05)
            
            # 장기간 거래 없이 홀드만 했다면 매수/매도 액션에 추가 보상
            if self.no_trade_steps > 10:
                reward += 0.1  # 오랜 정체기 후 거래에 보상
                
        elif action == 0:  # 매도
            # 과매수 상태나 상단 돌파 시 매도에 보상
            if rsi > 70 or current_price > bb_upper:
                reward += 0.2  # 매도 보상 증가 (이전 0.1)
            # 과매도 상태나 하단 돌파 시 매도에 페널티
            elif rsi < 30 or current_price < bb_lower:
                reward -= 0.1  # 페널티 감소 (이전 0.15)
            # 기본 매도 보상 추가
            else:
                reward += 0.05  # 기본 매도 보상 추가
            
            # 장기간 거래 없이 홀드만 했다면 매수/매도 액션에 추가 보상
            if self.no_trade_steps > 10:
                reward += 0.1  # 오랜 정체기 후 거래에 보상
            
        elif action == 1:  # 홀드에 대한 보상 감소
            # 적절한 조건에서 홀드 보상
            if 0.4 <= current_position_ratio <= 0.6:
                reward += 0.01  # 홀드 보상 감소 (이전 0.03)
            # 추세가 불확실할 때 홀드 보상
            if 40 <= rsi <= 60:
                reward += 0.01  # 홀드 보상 감소 (이전 0.02)
            
            # 오래 홀드만 지속하면 페널티 (10회 이상 연속 홀드 시)
            if self.no_trade_steps > 20:
                reward -= 0.001 * (self.no_trade_steps - 20)  # 과도한 홀드에 작은 페널티
        
        # 3. 포지션 관리 보상
        if 0.3 <= current_position_ratio <= 0.7:  # 적정 비율 조정 (이전 0.35~0.75)
            reward += 0.02  # 포지션 관리 보상 감소 (이전 0.05)
        # 극단적인 포지션 페널티 조정
        elif current_position_ratio > 0.95 or current_position_ratio < 0.05:
            reward -= 0.03  # 극단 페널티 감소 (이전 0.05)
        
        # 4. 거래 수수료 조정
        if trade_amount > 0:
            reward -= 0.001 * (trade_amount / total_value)  # 거래 수수료 감소 (이전 0.002)
        
        # 5. 과도한 거래 제한 완화
        if self.consecutive_actions >= 3:  # 4번 이상 연속 실행 시 페널티 (이전 3번)
            reward -= 0.05 * (self.consecutive_actions - 2)  # 페널티 감소 (이전 0.1)
        
        self.last_action = action
        
        info = {
            'portfolio_value': float(self.portfolio_value),
            'returns': float(returns),
            'action': int(action),
            'position_ratio': float(current_position_ratio),
            'rsi': float(rsi),
            'price': float(current_price),
            'shares': int(self.shares),
            'balance': float(self.balance)
        }
        
        return self._get_observation(), reward, done, truncated, info 