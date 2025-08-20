# Dockerfile for music_recommendation_app (수정 완료)

# --- 1단계: Node.js 클라이언트 준비 스테이지 ---
FROM node:18-slim as node-builder

# 수정: 작업 디렉토리 경로 변경
WORKDIR /app/getsongbpm_client

# 수정: 복사할 소스 경로 변경
COPY getsongbpm_client/package*.json ./
RUN npm install

# 수정: 복사할 소스 경로 변경
COPY getsongbpm_client/ ./


# --- 2단계: 최종 Python 애플리케이션 스테이지 ---
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성을 설치합니다.
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# 수정: 복사해 올 폴더 경로 변경
COPY --from=node-builder /app/getsongbpm_client ./getsongbpm_client

# Python 의존성을 설치합니다.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 Python 소스 코드를 복사합니다.
COPY . .

# 1. 수정: start.sh 스크립트에 실행 권한을 부여합니다.
RUN chmod +x ./start.sh

# 2. 수정: CMD 명령어를 start.sh 스크립트를 실행하도록 변경합니다.
CMD ["sh", "./start.sh"]