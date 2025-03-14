# 강화학습 기반 트레이딩 시스템

## 프로젝트 소개

이 프로젝트는 강화학습(Reinforcement Learning)을 활용하여 주식 시장에서 자동으로 매수, 매도, 홀드 결정을 내리는 트레이딩 시스템입니다. S&P 500 지수(^GSPC) 데이터를 기반으로 학습하며, 기술적 지표와 포트폴리오 상태를 분석하여 최적의 투자 결정을 내립니다.

## 시스템 구성

### 주요 파일
- **rl_env.py**: 강화학습 환경(Gymnasium 기반)을 정의합니다. 이 환경은 주식 데이터를 처리하고, 매수/매도/홀드 액션에 따른 보상을 계산합니다.
- **train_rl_agent.py**: 데이터 다운로드, 모델 학습, 평가 및 시각화 기능을 제공합니다.
- **strategies/**: 전통적인 트레이딩 전략(SMA 크로스오버, RSI, 볼린저 밴드)을 구현한 파일들이 포함되어 있습니다.

### 주요 기능
1. **데이터 처리**: Yahoo Finance에서 주식 데이터를 다운로드하고 전처리합니다.
2. **기술적 지표 계산**: RSI(상대강도지수), 볼린저 밴드 등의 기술적 지표를 계산합니다.
3. **강화학습 모델 학습**: PPO(Proximal Policy Optimization) 알고리즘을 사용하여 트레이딩 모델을 학습합니다.
4. **모델 평가**: 학습된 모델을 테스트 데이터에 적용하여 성능을 평가합니다.
5. **거래 결과 시각화**: 매수/매도/홀드 액션과 포트폴리오 가치 변화를 시각화합니다.

## 사용된 기술 및 알고리즘

### 주요 라이브러리
- **Gymnasium**: 강화학습 환경 구현
- **Stable-Baselines3**: PPO 알고리즘 구현
- **yfinance**: 주식 데이터 다운로드
- **pandas**: 데이터 처리
- **numpy**: 수치 연산
- **matplotlib**: 결과 시각화
- **backtrader**: 전통적 트레이딩 전략 테스트

### 알고리즘 및 접근 방식
- **PPO(Proximal Policy Optimization)**: 안정적인 정책 학습을 위한 최신 강화학습 알고리즘
- **기술적 지표 기반 의사결정**: RSI와 볼린저 밴드를 활용하여 시장 상황 평가
- **포트폴리오 밸런싱**: 주식 비율과 현금 비율을 조절하여 리스크 관리

## 설치 및 실행 방법

### 필수 요구사항
- Python 3.8 이상
- 필요 패키지: requirements.txt 참조

### 설치 방법
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요 패키지 설치
pip install -r requirements.txt
```

### 실행 방법
```bash
# 모델 학습 및 평가 실행
python train_rl_agent.py
```

## 주요 파라미터 설명

### 초기 투자금
- 기본값: 37,037 USD (약 5,000만원)
- 설정 방법: `rl_env.py`의 `PortfolioOptEnv` 클래스 초기화 시 `initial_balance` 매개변수 조정

### 학습 기간
- 기본값: 2015-01-01부터 2020-12-31까지
- 테스트 기간: 2021-01-01부터 2022-12-31까지
- 설정 방법: `train_rl_agent.py`의 `main()` 함수에서 `download_and_preprocess_data()` 호출 시 변경

### 강화학습 하이퍼파라미터
- 학습률(learning_rate): 5e-4
- 할인율(gamma): 0.95
- 엔트로피 계수(ent_coef): 0.1
- 네트워크 구조: [128, 64]
- 설정 방법: `train_rl_agent.py`의 `train_model()` 함수에서 변경

## 성능 평가 및 결과 해석

### 평가 지표
- **총 보상(reward)**: 모델이 얻은 누적 보상 점수
- **수익률(return)**: 초기 투자금 대비 최종 포트폴리오 가치의 변화율
- **액션 비율**: 매수/매도/홀드 결정의 비율
- **거래 빈도**: 전체 기간 대비 실제 거래(매수/매도)가 일어난 비율

### 결과 시각화
- 주가 차트 위에 매수/매도/홀드 액션을 표시
- 매수는 초록색 삼각형, 매도는 빨간색 역삼각형, 홀드는 파란색 원으로 표시
- 차트 하단에 액션 비율과 거래 통계 정보 출력

## 트레이딩 전략 설명

이 모델은 다음과 같은 전략적 접근을 취합니다:

1. **RSI 기반 매매**: RSI가 30 미만일 때는 매수 신호, 70 초과 시 매도 신호로 해석
2. **볼린저 밴드 활용**: 가격이 하단 밴드를 돌파하면 매수 신호, 상단 밴드를 돌파하면 매도 신호로 해석
3. **포지션 관리**: 전체 자산의 30%~70% 사이로 주식 비율을 유지하도록 관리
4. **과도한 거래 제한**: 연속적인 같은 액션에 페널티를 부여하여 과도한 거래 방지
5. **홀드 전략**: 시장이 불확실할 때(RSI 40~60)는 홀드를 선호

## 한계점 및 개선 방향

1. **더 다양한 지표 활용**: MACD, 이동평균선 등 추가 기술적 지표 도입
2. **멀티 에셋 지원**: 여러 종목을 동시에 트레이딩하는 포트폴리오 최적화
3. **뉴스 데이터 통합**: 시장 뉴스와 감성 분석을 모델에 통합
4. **리스크 관리 강화**: 변동성 기반 포지션 사이징 및 손절매 전략 구현
5. **하이퍼파라미터 최적화**: 베이지안 최적화 등을 통한 하이퍼파라미터 튜닝

## 주의사항

이 모델은 교육 및 연구 목적으로 개발되었으며, 실제 투자에 직접 활용할 경우 금전적 손실이 발생할 수 있습니다. 항상 투자의 위험성을 인지하고, 전문가의 조언을 구하시기 바랍니다. 