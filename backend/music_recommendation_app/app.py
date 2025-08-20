import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from dotenv import load_dotenv # 1. 라이브러리 임포트

from music_recommender import MusicRecommender

# --- 1. 앱 초기화 및 로깅 설정 ---
app = Flask(__name__)
load_dotenv() # 2. .env 파일을 로드하여 환경 변수로 설정

CORS(app, resources={r"/recommend": {"origins": "*"}})
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. 모델 사전 로딩 (Pre-loading) ---
recommender = None
try:
    logging.info("Music Recommendation App API 서버 초기화 시작...")
    
    GETSONGBPM_API_KEY = os.environ.get("GETSONGBPM_API_KEY")
    if not GETSONGBPM_API_KEY:
        # API 키가 없으면 서버가 시작은 되지만, 추천 시 오류를 반환하게 됩니다.
        logging.warning("GETSONGBPM_API_KEY 환경 변수가 설정되지 않았습니다. API 호출이 제한됩니다.")
    
    # 이 시점에서 MusicRecommender가 초기화되며, 내부의 SentimentAnalyzer가 모델을 로드합니다.
    recommender = MusicRecommender(GETSONGBPM_API_KEY)
    
    logging.info("추천기 객체(MusicRecommender) 초기화 및 모델 로딩 완료.")

except Exception as e:
    logging.critical(f"치명적 오류: 추천기 객체 초기화 중 실패: {e}", exc_info=True)
    recommender = None # 초기화 실패를 명확히 함

# --- 3. API 엔드포인트 정의 ---
@app.route('/healthz')
def health_check():
    """헬스 체크 엔드포인트. 서비스의 준비 상태를 확인합니다."""
    if recommender:
        return "OK", 200
    else:
        # 503 Service Unavailable: 서비스가 현재 요청을 처리할 준비가 안 됨
        return "Recommender not initialized", 503

@app.route('/recommend', methods=['POST'])
def recommend_music_endpoint():
    """
    사용자 메시지를 받아 음악을 추천하고 JSON으로 결과를 반환하는 API 엔드포인트.
    """
    if not recommender:
        logging.error("API 오류: 추천기 객체가 초기화되지 않았습니다.")
        return jsonify({"error": "Service is currently unavailable. Recommender is not initialized."}), 503

    user_text = request.form.get('user_message')
    
    if not user_text or not user_text.strip():
        logging.warning("API 경고: 'user_message'가 비어있습니다.")
        return jsonify({"error": "Please provide your emotion or situation in 'user_message'."}), 400

    logging.info(f"API 요청 수신: '{user_text}'에 대한 음악 추천 시작")
    
    try:
        # recommender.recommend_music은 이제 최종 결과 딕셔너리를 반환합니다.
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

# --- 4. 로컬 개발 환경에서 직접 실행 ---
if __name__ == '__main__':
    # Render와 같은 배포 환경에서는 Gunicorn이 이 파일을 실행하므로, 아래 코드는 실행되지 않습니다.
    # 로컬에서 `python app.py`를 실행할 때만 사용됩니다.
    port = int(os.environ.get("PORT", 5001))
    # debug=False로 설정하여, 파일 변경 시 서버가 재시작되어 모델을 다시 로드하는 것을 방지합니다.
    app.run(host='0.0.0.0', port=port, debug=False)