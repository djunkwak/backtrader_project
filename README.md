# Backtrader 프로젝트

이 프로젝트는 Backtrader를 사용한 간단한 트레이딩 전략 백테스팅 예제입니다.

## 설치 방법

1. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# 또는
.\venv\Scripts\activate  # Windows
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

1. `simple_strategy.py` 실행:
```bash
python simple_strategy.py
```

이 예제는 AAPL 주식에 대한 이동평균 크로스오버 전략을 백테스팅합니다.

## 주요 기능

- 단기/장기 이동평균 크로스오버 전략 구현
- Yahoo Finance에서 실제 주가 데이터 사용
- 거래 수수료 설정
- 백테스팅 결과 시각화

## 커스터마이징

- `fast_period`와 `slow_period` 파라미터를 조정하여 이동평균 기간 변경 가능
- 다른 종목 심볼로 변경하여 테스트 가능
- 거래 수수료 조정 가능 