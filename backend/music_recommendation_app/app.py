# app.py (최종 수정안)
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time

# MusicRecommender 클래스를 임포트합니다.
from music_recommender import MusicRecommender

# --- 1. 앱 초기화 및 로깅 설정 ---
app = Flask(__name__)
CORS(app, resources={r"/recommend": {"origins": "*"}})
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. 모델 사전 로딩 (Pre-loading) ---
# 서버 시작 시 이 코드가 실행되며, MusicRecommender 내부의 모델 로딩이 시작됩니다.
# 이 과정은 매우 오래 걸릴 수 있으며, 이는 정상입니다.
recommender = None
try:
    logging.info("Music Recommendation App API 서버 초기화 시작...")
    start_time = time.time()
    
    GETSONGBPM_API_KEY = os.environ.get("GETSONGBPM_API_KEY")
    if not GETSONGBPM_API_KEY:
        logging.warning("GETSONGBPM_API_KEY 환경 변수가 설정되지 않았습니다. API 호출이 제한됩니다.")
    
    # 이 시점에서 MusicRecommender의 __init__이 호출되며, 모델 로딩이 시작됩니다.
    recommender = MusicRecommender(GETSONGBPM_API_KEY)
    
    end_time = time.time()
    loading_time = end_time - start_time
    logging.info(f"추천기 객체(MusicRecommender) 초기화 및 모델 로딩 완료. (소요 시간: {loading_time:.2f}초)")

except Exception as e:
    logging.critical(f"치명적 오류: 추천기 객체 초기화 중 실패: {e}", exc_info=True)
    recommender = None # 초기화 실패를 명확히 함

# --- 3. API 엔드포인트 (기존과 거의 동일) ---
@app.route('/healthz')
def health_check():
    """헬스 체크 엔드포인트. 서비스 상태를 확인합니다."""
    if recommender:
        return "OK", 200
    else:
        # 503 Service Unavailable: 서비스가 현재 요청을 처리할 준비가 안 됨
        return "Recommender not initialized", 503

@app.route('/recommend', methods=['POST'])
def recommend_music_endpoint():
    """사용자 메시지를 받아 음악을 추천하고 JSON으로 결과를 반환하는 API 엔드포인트."""
    if not recommender:
        logging.error("API 오류: 추천기 객체가 초기화되지 않았습니다.")
        return jsonify({"error": "Service is currently unavailable. Recommender is not initialized."}), 503

    user_text = request.form.get('user_message')
    if not user_text or not user_text.strip():
        logging.warning("API 경고: 'user_message'가 비어있습니다.")
        return jsonify({"error": "Please provide your emotion or situation in 'user_message'."}), 400

    logging.info(f"API 요청 수신: '{user_text}'에 대한 음악 추천 시작")
    
    try:
        # 이제 이 메소드는 모델을 로드하지 않으므로 매우 빠르게 실행됩니다.
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
    # 디버그 모드는 프로덕션에서 False여야 합니다. 
    # True로 설정 시 파일 변경 시 서버가 재시작되어 모델을 다시 로드합니다.
    app.run(host='0.0.0.0', port=5001, debug=False)