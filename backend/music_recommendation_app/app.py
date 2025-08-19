# app.py (수정 완료)
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# 표준 상대 경로 임포트
# MockMusicRecommender는 삭제되었으므로, MusicRecommender만 임포트합니다.
from .music_recommender import MusicRecommender

# Flask 앱 인스턴스 생성
app = Flask(__name__)

# CORS 설정
CORS(app, resources={r"/recommend": {"origins": "*"}})

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Music Recommendation App API 서버 초기화 시작.")

# 추천기 인스턴스 생성
recommender = None
try:
    # GETSONGBPM_API_KEY가 없으면 MusicRecommender 내부에서 처리하므로, 여기서 바로 생성합니다.
    GETSONGBPM_API_KEY = os.environ.get("GETSONGBPM_API_KEY")
    if not GETSONGBPM_API_KEY:
        logging.warning("GETSONGBPM_API_KEY 환경 변수가 설정되지 않았습니다. API 호출이 제한됩니다.")
    
    recommender = MusicRecommender(GETSONGBPM_API_KEY)
    logging.info("추천기 객체(MusicRecommender)가 성공적으로 초기화되었습니다.")

except Exception as e:
    logging.critical(f"추천기 객체 초기화 실패: {e}", exc_info=True)
    recommender = None # 초기화 실패 시 recommender를 None으로 설정

@app.route('/healthz')
def health_check():
    """헬스 체크 엔드포인트. 서비스 상태를 확인합니다."""
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
    app.run(host='0.0.0.0', port=port, debug=False)