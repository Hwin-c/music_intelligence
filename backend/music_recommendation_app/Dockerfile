# ======================================================================
# Music Genre API - Dockerfile (최종 수정안)
# 성공한 music_recommendation_api와 동일한 단일 스테이지 빌드 방식으로 단순화하여
# Render 플랫폼과의 호환성 문제를 해결합니다.
# ======================================================================

# --- 최종 애플리케이션 스테이지 ---
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성을 설치합니다.
# build-essential은 pip 설치 중 C 확장 모듈 컴파일에 필요할 수 있으며,
# ffmpeg은 오디오 처리에 필수적입니다.
RUN apt-get update && apt-get install -y build-essential ffmpeg && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 의존성 설치
# 이 단계에서 최적화된 필수 라이브러리만 설치됩니다.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# start.sh 스크립트에 실행 권한 부여
RUN chmod +x ./start.sh

# start.sh 스크립트를 실행하여 서버 시작 (sh를 명시하여 안정성 확보)
CMD ["sh", "./start.sh"]