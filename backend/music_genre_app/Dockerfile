# Dockerfile for music_genre_app on Fly.io

# 1. 기반 이미지 설정: Python 3.11 슬림 버전을 사용합니다.
# 'slim' 버전은 배포에 불필요한 패키지를 제외하여 이미지 크기를 줄여줍니다.
FROM python:3.11-slim

# 2. 작업 디렉토리 설정: 컨테이너 내에서 /app 폴더를 기본 경로로 사용합니다.
WORKDIR /app

# 3. 시스템 의존성 설치: Python 라이브러리가 필요로 하는 시스템 레벨의 도구들을 설치합니다.
#    - build-essential: C/C++ 코드를 컴파일하는 데 필요한 도구 모음입니다. ('cc' 컴파일러 포함)
#    - ffmpeg: audioread/librosa가 오디오 파일을 처리하는 데 사용하는 프로그램입니다.
RUN apt-get update && apt-get install -y build-essential ffmpeg && rm -rf /var/lib/apt/lists/*

# 4. Python 의존성 설치: requirements.txt 파일을 먼저 복사하여 라이브러리를 설치합니다.
#    이 순서는 소스 코드만 변경되었을 때, 매번 라이브러리를 다시 설치하는 것을 방지하여 빌드 속도를 높입니다.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사: 현재 폴더(.)의 모든 파일(app.py, model.keras 등)을 컨테이너의 /app 폴더로 복사합니다.
COPY . .

# 6. 서버 실행 명령어: 컨테이너가 시작될 때 Gunicorn 웹 서버를 실행합니다.
#    - --bind :$PORT : Fly.io가 동적으로 할당하는 포트 번호를 사용합니다. (보통 8080)
#    - --workers 1 --threads 8 : Fly.io의 기본 CPU(shared-cpu-1x)에 적합한 설정입니다.
#    - --timeout 0 : 모델 로딩 등 시간이 오래 걸리는 작업이 타임아웃되지 않도록 설정합니다.
#    - app:app : 'app.py' 파일 안에 있는 'app'이라는 이름의 Flask 객체를 실행하라는 의미입니다.
# --no-preload: 각 워커가 독립적으로 앱을 로드하도록 하여 시작 부담을 분산시킵니다.
# --graceful-timeout 60: 워커가 시작되고 응답할 때까지 60초간 기다려줍니다.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 --no-preload --graceful-timeout 60 app:app