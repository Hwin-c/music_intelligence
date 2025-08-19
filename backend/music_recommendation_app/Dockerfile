# Dockerfile for music_recommendation_app

# --- 1단계: Node.js 클라이언트 빌드 스테이지 ---
# Node.js 18-slim 버전을 'node-builder'라는 별명의 임시 컨테이너로 사용합니다.
FROM node:18-slim as node-builder

# 작업 디렉토리를 설정합니다.
WORKDIR /app/getsongbpm-client

# 먼저 package.json 파일을 복사하여 의존성을 설치합니다.
COPY getsongbpm-client/package*.json ./
RUN npm install

# 나머지 getsongbpm-client 소스 코드를 복사합니다.
COPY getsongbpm-client/ ./


# --- 2단계: 최종 Python 애플리케이션 스테이지 ---
# Python 3.9 슬림 버전을 최종 배포 이미지의 기반으로 사용합니다.
FROM python:3.9-slim

# 작업 디렉토리를 설정합니다.
WORKDIR /app

# 1단계(node-builder)에서 설치하고 준비된 getsongbpm-client 폴더 전체를 복사해옵니다.
COPY --from=node-builder /app/getsongbpm-client ./getsongbpm-client

# Python 의존성을 설치합니다.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 Python 소스 코드(app.py, music_recommender.py 등)를 복사합니다.
COPY . .

# Gunicorn 웹 서버로 Flask 앱을 실행합니다.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app