import os
from flask import Flask, render_template, request, jsonify
import logging
import sys
import json # json 모듈 임포트

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Music Recommendation App 초기화 시작.")

# music_recommender.py가 있는 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    logging.debug(f"Added {current_dir} to sys.path for music_recommender.py")

from music_recommender import MusicRecommender, MockMusicRecommender

app = Flask(__name__)

# 환경 변수에서 API 키 로드
GETSONGBPM_API_KEY = os.environ.get("GETSONGBPM_API_KEY")

# API 키 유무에 따라 실제 또는 Mock 추천기 초기화
if GETSONGBPM_API_KEY:
    logging.info("GETSONGBPM_API_KEY가 설정되어 실제 API를 사용 합니다.")
    recommender = MusicRecommender(GETSONGBPM_API_KEY)
else:
    logging.warning("GETSONGBPM_API_KEY 환경 변수가 설정되지 않았습니다. Mock 추천기를 사용합니다.")
    recommender = MockMusicRecommender()

logging.info("음악 추천 모델 및 도구 로드 성공.")

@app.route('/')
def index():
    """
    메인 페이지를 렌더링합니다.
    """
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_music_endpoint():
    """
    사용자 메시지를 받아 음악을 추천하고 결과를 반환합니다.
    """
    user_text = request.form.get('user_message')
    if not user_text:
        return jsonify({"error": "No user message provided"}), 400

    logging.info(f"\n--- Starting music recommendation for '{user_text}' ---")
    
    try:
        # recommend_music 함수가 이제 딕셔너리를 반환합니다.
        recommendation_data = recommender.recommend_music(user_text)
        
        # 결과를 recommend_loading_page.html로 전달합니다.
        # recommendation_data는 딕셔너리이므로, JSON 문자열로 변환하여 전달합니다.
        return render_template(
            'recommend_loading_page.html',
            user_message=user_text,
            recommendation_info_json=json.dumps(recommendation_data) # JSON 문자열로 변환
        )
    except Exception as e:
        logging.error(f"음악 추천 중 오류 발생: {e}")
        # 오류 발생 시 사용자에게 친화적인 메시지 표시
        # 이 경우에도 recommendation_info_json을 빈 값으로 전달하여 JS 오류 방지
        return render_template(
            'recommend_loading_page.html', # 로딩 페이지에서 오류 메시지를 보여주거나, 바로 결과 페이지로 넘어가도록
            user_message=user_text,
            recommendation_info_json=json.dumps({"recommendations": [], "user_emotion": "오류 발생", "target_audio_features": {"bpm": (0,0), "danceability": (0,0), "acousticness": (0,0)}, "error_message": "음악을 추천하는 중 오류가 발생했습니다. 다시 시도해주세요."})
        )

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
            recommendation_info = {"recommendations": [], "user_emotion": "알 수 없음", "target_audio_features": {"bpm": (0,0), "danceability": (0,0), "acousticness": (0,0)}}
    
    # recommendation_info가 비어있을 경우 기본값 설정 (오류 방지)
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
    app.run(debug=True)