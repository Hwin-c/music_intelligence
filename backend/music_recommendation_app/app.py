# app.py (수정 완료)
import os
from flask import Flask, request, jsonify
from flask_cors import CORS # CORS 라이브러리 임포트
import logging

# ======================================================================
# 1. 표준 상대 경로 임포트로 변경 (sys.path 조작 제거)
# ======================================================================
# '.'은 현재 패키지(music_recommendation_app)를 의미합니다.
from .music_recommender import MusicRecommender, MockMusicRecommender
# ======================================================================

# Flask 앱 인스턴스 생성 (API 서버이므로 static/template 폴더 설정 불필요)
app = Flask(__name__)

# ======================================================================
# 2. CORS 설정 (Vercel 프론트엔드와의 통신을 위해 필수)
# ======================================================================
CORS(app, resources={r"/recommend": {"origins": "*"}})
logging.info("CORS 설정 완료. /recommend 엔드포인트에 대해 모든 출처의 요청을 허용합니다.")
# ======================================================================

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Music Recommendation App API 서버 초기화 시작.")

# 추천기 인스턴스 생성
recommender = None
try:
    GETSONGBPM_API_KEY = os.environ.get("GETSONGBPM_API_KEY")
    if GETSONGBPM_API_KEY:
        logging.info("GETSONGBPM_API_KEY가 설정되어 실제 API를 사용하는 추천기를 초기화합니다.")
        recommender = MusicRecommender(GETSONGBPM_API_KEY)
    else:
        logging.warning("GETSONGBPM_API_KEY 환경 변수가 설정되지 않았습니다. Mock 추천기를 사용합니다.")
        recommender = MockMusicRecommender()
    logging.info(f"추천기 객체 타입: {type(recommender).__name__}")
except Exception as e:
    logging.critical(f"추천기 객체 초기화 실패: {e}", exc_info=True)
    recommender = None # 초기화 실패 시 recommender를 None으로 설정

@app.route('/healthz')
def health_check():
    """헬스 체크 엔드포인트. 서비스 상태를 확인합니다."""
    # 추천기 객체가 성공적으로 로드되었는지 확인
    if recommender:
        return "OK", 200
    else:
        return "Recommender not initialized", 503

@app.route('/recommend', methods=['POST'])
def recommend_music_endpoint():
    """사용자 메시지를 받아 음악을 추천하고 JSON으로 결과를 반환하는 API 엔드포인트."""
    if not recommender:
        logging.error("API 오류: 추천기 객체가 초기화되지 않았습니다.")
        return jsonify({"error": "Service is currently unavailable. Recommender not initialized."}), 503

    user_text = request.form.get('user_message')
    if not user_text or not user_text.strip():
        logging.warning("API 경고: 'user_message'가 비어있습니다.")
        return jsonify({"error": "Please provide your emotion or situation in 'user_message'."}), 400

    logging.info(f"API 요청 수신: '{user_text}'에 대한 음악 추천 시작")
    
    try:
        recommendation_info = recommender.recommend_music(user_text)
        
        # 프론트엔드가 사용하기 좋은 형태로 최종 JSON 응답 구성
        response_data = {
            "user_message": user_text,
            "recommendation_info": recommendation_info
        }
        logging.info(f"추천 성공: '{user_text}' -> 감정: {recommendation_info.get('user_emotion')}")
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"API 오류: 음악 추천 중 예외 발생 - {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred during recommendation: {str(e)}"}), 500

# 로컬 개발 환경에서 직접 실행할 때 사용됩니다.
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    # Gunicorn이 프로덕션에서 실행하므로, 로컬 테스트 시 debug=False로 설정
    app.run(host='0.0.0.0', port=port, debug=False)