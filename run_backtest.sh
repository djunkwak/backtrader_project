#!/bin/bash

# 프로젝트 디렉토리로 이동
cd "$(dirname "$0")"

# 가상환경이 이미 존재하는지 확인
if [ ! -d "venv" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
source venv/bin/activate

# 필요한 패키지 설치
echo "필요한 패키지 설치 중..."
pip install backtrader
pip install yfinance
pip install pandas
pip install matplotlib

# 백테스팅 스크립트 실행
echo "백테스팅 실행 중..."
python simple_strategy.py

# 가상환경 비활성화
deactivate
