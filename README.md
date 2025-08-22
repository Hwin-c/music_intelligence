# 🎵 PickNFit (음잘딱깔센)

**프로젝트명:** PickNFit (음잘딱깔센) - 인공지능이 음악을 잘 딱 깔끔하고 센스 있게 장르 분류 및 추천해준다는 뜻을 내포하고 있습니다.

## ✨ 프로젝트 내용

이 프로젝트는 크게 두 가지 핵심 기능을 가진 웹 서비스를 구현하고 배포하는 것을 목표로 합니다.

1. **음악 장르 분류:** 사용자가 오디오 파일(`.wav`)을 업로드하면, 인공지능 모델이 파일을 분석하여 10가지 음악 장르 중 하나로 분류합니다.  
2. **감정 기반 음악 추천:** 사용자의 자연어 입력을 60가지 감정 중 하나로 분석하고, 이를 바탕으로 음악적 특성을 고려한 개인화된 음악을 추천합니다.

**배포 환경:**  
이 서비스는 프론트엔드와 백엔드가 분리된 MSA(Microservice Architecture) 구조로 배포되었습니다.
- **Frontend:** Vercel을 통해 사용자 인터페이스 제공  
- **Backend:** Render를 통해 두 개의 독립적인 API 서버(장르 분류, 음악 추천) 운영  
- **참고:** Render Free 티어의 리소스 제약으로 인해, 배포된 추천 서비스는 2가지 감정(긍정/부정)으로 단순화된 모델을 사용 중입니다. 로컬 환경에서는 60가지 감정 모델이 정상 작동합니다.

**배포된 웹사이트:**  
👉 [https://pick-n-fit.vercel.app/](https://pick-n-fit.vercel.app/)

---

## 🎬 구현 영상
(추후 추가 예정)

---

## 🚀 로컬 환경 설정 및 실행 방법

이 프로젝트는 **Python Flask 프레임워크**와 **순수 HTML/CSS/JS**로 개발되었습니다.  
아래 지침에 따라 로컬 환경을 설정하고 애플리케이션을 실행할 수 있습니다.

### 📋 개발 환경
- **Python 3.11.9** 이상  
- [Git](https://git-scm.com/downloads)

---

### ⬇️ 프로젝트 클론
```bash
git clone https://github.com/Hwin-c/music_intelligence.git
cd music_intelligence
````

---

### 📦 의존성 설치

가상 환경을 생성하고, 각 백엔드 서비스에 필요한 라이브러리를 `requirements.txt` 파일을 통해 설치합니다.

가상 환경 생성 및 활성화:

```bash
# 프로젝트 루트 디렉토리에서 실행
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS/Linux (Bash)
source venv/bin/activate
```

필요한 라이브러리 설치:

```bash
# (venv)가 활성화된 상태에서 실행
pip install -r backend/music_genre_app/requirements.txt
pip install -r backend/music_recommendation_app/requirements.txt
```

---

### 🔑 API Key 설정

음악 추천 기능에는 **GetsongBPM API 키**가 필요합니다. `.env` 파일을 사용하여 API 키를 안전하게 관리합니다.

1. `.env` 파일 생성

   * `backend/music_recommendation_app/` 디렉토리 내에 `.env` 파일을 새로 만듭니다.
2. API 키 추가

   * 생성한 `.env` 파일에 아래 내용을 추가하고, 실제 발급받은 키로 교체합니다.

```env
GETSONGBPM_API_KEY="YOUR_GETSONGBPM_API_KEY"
```

👉 참고: `.env` 파일은 `.gitignore`에 등록되어 있어 Git 저장소에 포함되지 않습니다.

---

### 🖥️ 애플리케이션 실행

이 프로젝트는 \*\*3개의 서버(프론트엔드 1개, 백엔드 2개)\*\*로 구성되며, 각각 별도의 터미널에서 실행해야 합니다.
(모든 터미널에서 가상환경이 활성화되어 있어야 합니다.)

#### 1️⃣ 프론트엔드 서버 ([http://127.0.0.1:8000](http://127.0.0.1:8000))

```bash
# 프로젝트 루트 디렉토리(music_intelligence/)에서 실행
python -m http.server 8000 --directory frontend
```

#### 2️⃣ 음악 장르 분류 API ([http://127.0.0.1:5000](http://127.0.0.1:5000))

```bash
cd backend/music_genre_app
python app.py
```

#### 3️⃣ 감정 기반 음악 추천 API ([http://127.0.0.1:5001](http://127.0.0.1:5001))

```bash
cd backend/music_recommendation_app
python app.py
```

👉 모든 서버가 실행되면 웹 브라우저에서 [http://127.0.0.1:8000](http://127.0.0.1:8000) 으로 접속하여 서비스를 이용할 수 있습니다.

---

## 📄 라이선스

이 프로젝트는 **MIT License**를 따릅니다.

```