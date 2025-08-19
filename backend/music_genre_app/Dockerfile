# Dockerfile for music_genre_app on Fly.io (Multi-stage build for optimization)

# --- 1단계: 빌더(Builder) 스테이지 ---
# 여기서는 필요한 모든 도구를 설치하고 라이브러리를 빌드합니다.
FROM python:3.11 as builder

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y build-essential ffmpeg && rm -rf /var/lib/apt/lists/*

# 가상환경 생성 및 활성화
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# requirements.txt 복사 및 의존성 설치
# --no-cache-dir을 사용하여 pip 캐시를 남기지 않습니다.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# --- 2단계: 최종(Final) 스테이지 ---
# 여기서는 실행에 필요한 최소한의 것들만 빌더 스테이지에서 가져옵니다.
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 중 '실행'에만 필요한 ffmpeg만 다시 설치합니다.
# build-essential 같은 '빌드'용 도구는 최종 이미지에 포함하지 않아 용량을 줄입니다.
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 빌더 스테이지에서 설치된 Python 가상환경(모든 라이브러리 포함)을 통째로 복사합니다.
COPY --from=builder /opt/venv /opt/venv

# 가상환경을 활성화합니다.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 소스 코드 복사
COPY . .

# 1. 수정: start.sh 스크립트에 실행 권한을 부여합니다.
RUN chmod +x ./start.sh

# 2. 수정: CMD 명령어를 start.sh 스크립트를 실행하도록 변경합니다.
CMD ["./start.sh"]