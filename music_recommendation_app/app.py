from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import logging
import traceback
import sys
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Music Recommendation App 초기화 시작.")

# music_recommender.py 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    logging.debug(f"Added {current_dir} to sys.path for music_recommender.py")

# music_recommender.py에서 MusicRecommender와 MockMusicRecommender 클래스를 임포트
from music_recommender import MusicRecommender, MockMusicRecommender

app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# Getsong API 키 환경 변수
GETSONGBPM_API_KEY = os.environ.get("GETSONGBPM_API_KEY")
# Getsong API의 텍스트 기반 추천 엔드포인트 URL을 위한 새로운 환경 변수
GETSONG_RECOMMENDATION_API_URL = os.environ.get("GETSONG_RECOMMENDATION_API_URL")

# API 키와 URL이 모두 설정되어 있으면 실제 MusicRecommender를 사용하고, 그렇지 않으면 Mock을 사용
if GETSONGBPM_API_KEY and GETSONG_RECOMMENDATION_API_URL:
    logging.info("GETSONGBPM_API_KEY와 GETSONG_RECOMMENDATION_API_URL이 설정되어 실제 API를 사용합니다.")
    recommender = MusicRecommender(GETSONGBPM_API_KEY, GETSONG_RECOMMENDATION_API_URL)
else:
    logging.warning("GETSONGBPM_API_KEY 또는 GETSONG_RECOMMENDATION_API_URL 환경 변수가 설정되지 않았습니다. Mock 추천기를 사용합니다.")
    # Mock 추천기에도 URL 인자를 전달하여 일관성을 유지합니다.
    recommender = MockMusicRecommender(GETSONGBPM_API_KEY, GETSONG_RECOMMENDATION_API_URL)

logging.info(f"Recommender 객체 타입: {type(recommender).__name__}")
logging.info("음악 추천 모델 및 도구 로드 성공.")

@app.route('/')
@app.route('/recommendation') # /recommendation 경로도 index.html을 렌더링
def recommendation_index():
    logging.info("음악 추천 앱 메인 페이지 요청 수신.")
    # 오류 메시지가 있다면 함께 렌더링
    error_message = request.args.get('error_message')
    return render_template('index.html', error_message=error_message)

@app.route('/healthz')
def health_check():
    """Render 헬스 체크를 위한 엔드포인트."""
    logging.debug("Health check 요청 수신.")
    return "OK", 200

@app.route('/recommend', methods=['POST'])
def recommend_music_endpoint():
    """
    사용자 메시지를 받아 음악을 추천하고 결과를 반환합니다.
    """
    user_text = request.form.get('user_message')
    
    if not user_text or user_text.strip() == '':
        logging.warning("User message is empty. Returning 400 Bad Request.")
        # 클라이언트 JavaScript에서 이 JSON 응답을 받아서 처리할 것입니다.
        return jsonify({"error": "감정을 입력해주세요!"}), 400

    logging.info(f"\n--- Starting music recommendation for '{user_text}' ---")
    
    try:
        # recommender 객체에 recommend_music 메서드가 있어야 합니다.
        recommendation_info = recommender.recommend_music(user_text)
        
        # Flask는 JSON 응답을 직접 보냅니다.
        # 이 응답은 index.html의 fetch 요청에서 받게 됩니다.
        return jsonify({
            "user_message": user_text, # 원본 사용자 메시지 포함
            "recommendation_info": recommendation_info # 추천 정보 전체 포함
        })
    except Exception as e:
        logging.error(f"음악 추천 중 오류 발생: {e}", exc_info=True)
        # 오류 발생 시 JSON 응답으로 오류 메시지 전달
        return jsonify({"error": f"음악을 추천하는 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/recommend_result')
def recommend_result_page():
    """
    음악 추천 결과를 표시하는 페이지를 렌더링합니다.
    """
    user_message = request.args.get('user_message')
    recommendation_info_json = request.args.get('recommendation_info_json')
    
    recommendation_info = {}
    if recommendation_info_json:
        try:
            recommendation_info = json.loads(recommendation_info_json)
        except json.JSONDecodeError:
            logging.error("Failed to decode recommendation_info JSON string.")
            # 파싱 오류 시 기본값 설정
            recommendation_info = {"recommendations": [], "user_emotion": "알 수 없음", "target_audio_features": {"bpm": (0,0), "danceability": (0,0), "acousticness": (0,0)}}
    
    # 필수 필드가 없는 경우 기본값 설정
    if not recommendation_info.get("recommendations"):
        recommendation_info["recommendations"] = []
    if not recommendation_info.get("user_emotion"):
        recommendation_info["user_emotion"] = "알 수 없음"
    if not recommendation_info.get("target_audio_features"):
        recommendation_info["target_audio_features"] = {"bpm": (0,0), "danceability": (0,0), "acousticness": (0,0)}

    return render_template(
        'recommend_result_page.html',
        user_message=user_message,
        recommendation_info=recommendation_info
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)