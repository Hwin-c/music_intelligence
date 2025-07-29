# 🎵 음잘딱깔센

**프로젝트명:** 음잘딱깔센 (인공지능이 음악을 잘 딱 깔끔하고 센스 있게 장르 분류 및 추천해준다는 뜻이 내포되어 있음)

## ✨ 프로젝트 내용

이 프로젝트는 크게 두 가지 핵심 기능을 가진 웹사이트를 구현하고 배포하는 것을 목표로 합니다.

1. **음악 장르 분류 기능:** 사용자가 오디오 파일을 업로드하면, 인공지능 모델이 해당 파일을 분석하여 음악 장르를 분류해줍니다.  
2. **감성 기반 음악 추천 기능:** 사용자의 자연어 입력을 감정 분석하고, 이를 바탕으로 개인화된 음악을 추천해줍니다.

**로컬 환경 특성:**  
로컬 개발 환경에서는 위 두 기능을 모두 통합한 하나의 웹사이트로 구현되어 프로젝트가 의도한 완전한 모습을 보여줍니다.  
반면, **배포 환경에서는 리소스 제약으로 인해 두 기능이 분리된 웹사이트로 운영**됩니다.

## 🚀 로컬 환경 설정 및 실행 방법

이 프로젝트는 Python Flask 프레임워크를 사용하여 개발되었습니다.  
아래 지침에 따라 로컬 환경을 설정하고 애플리케이션을 실행할 수 있습니다.

### 📋 개발 환경

- Python **3.11.9**
- [Git](https://git-scm.com/downloads)

### ⬇️ 프로젝트 클론

먼저 GitHub 리포지토리를 클론합니다:

```bash
git clone <YOUR_REPOSITORY_URL>
cd music_intelligence
```

### 📦 의존성 설치

가상 환경 사용을 권장하며, 아래 순서로 설치합니다:

1. **가상 환경 생성 및 활성화:**

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. **필요한 라이브러리 설치:**

```bash
pip install Flask requests librosa transformers torch
# 또는 TensorFlow 기반 사용 시:
# pip install Flask requests librosa transformers tensorflow
```

- `Flask`: 웹 프레임워크  
- `requests`: 외부 API 호출  
- `librosa`: 오디오 특징 추출  
- `transformers`, `torch` or `tensorflow`: 감성 분석 모델 구동

### 🔑 API Key 설정

음악 추천 기능에는 GetsongBPM API 키가 필요합니다. `GETSONGBPM_API_KEY` 환경 변수를 설정하세요.

- **Windows (PowerShell):**

```bash
$env:GETSONGBPM_API_KEY="YOUR_GETSONGBPM_API_KEY"
```

- **macOS/Linux (Bash):**

```bash
export GETSONGBPM_API_KEY="YOUR_GETSONGBPM_API_KEY"
```

> `YOUR_GETSONGBPM_API_KEY`는 실제 발급받은 키로 교체해야 합니다.

### 🖥️ 애플리케이션 실행

이 프로젝트는 3개의 Flask 서버로 구성되며, 각각 **별도 터미널에서 실행**해야 합니다.

#### 1️⃣ 메인 앱 (`http://127.0.0.1:8000`)

```bash
cd C:\Users\USER\Desktop\music_intelligence
set FLASK_APP=app.py
flask run --port 8000
```

#### 2️⃣ 음악 장르 분류 앱 (`http://127.0.0.1:5000`)

```bash
cd C:\Users\USER\Desktop\music_intelligence\music_genre_app
set FLASK_APP=app.py
flask run --port 5000
```

#### 3️⃣ 감성 기반 음악 추천 앱 (`http://127.0.0.1:5001`)

```bash
cd C:\Users\USER\Desktop\music_intelligence\music_recommendation_app
set FLASK_APP=app.py
flask run --port 5001
```

모든 서버가 실행되면 웹 브라우저에서 `http://127.0.0.1:8000`으로 접속하세요.

## 🎬 구현 영상

프로젝트의 주요 기능을 보여주는 데모 영상 링크를 여기에 삽입하세요:  
예시: [https://www.youtube.com/watch?v=YOUR_VIDEO_ID](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

## 📄 라이선스

이 프로젝트는 [MIT License](https://opensource.org/licenses/MIT)를 따릅니다.