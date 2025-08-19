# Dockerfile for music_genre_app

# 1. 기반 이미지 설정: Python 3.11 슬림 버전을 사용합니다.
FROM python:3.11-slim

# 2. 작업 디렉토리 설정: 컨테이너 내에서 /app 폴더를 기본 경로로 사용합니다.
WORKDIR /app

# 3. 의존성 설치: requirements.txt 파일을 먼저 복사하여 라이브러리를 설치합니다.
#    이렇게 하면 소스 코드가 변경될 때마다 라이브러리를 다시 설치하는 것을 방지하여 빌드 속도를 높일 수 있습니다.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --compile --no-binary=:all:

# 4. 소스 코드 복사: 현재 폴더(.)의 모든 파일(app.py, model.keras 등)을 컨테이너의 /app 폴더로 복사합니다.
COPY . .

# 5. 서버 실행 명령어: 컨테이너가 시작될 때 Gunicorn 웹 서버를 실행합니다.
#    - --bind :$PORT : Google Cloud Run이 동적으로 할당하는 포트 번호를 사용합니다.
#    - --workers 1 --threads 8 : Cloud Run의 기본 CPU(1 vCPU)에 적합한 설정입니다.
#    - --timeout 0 : 모델 로딩 등 시간이 오래 걸리는 작업이 타임아웃되지 않도록 설정합니다.
#    - app:app : 'app.py' 파일 안에 있는 'app'이라는 이름의 Flask 객체를 실행하라는 의미입니다.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app