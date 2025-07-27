import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import logging
import sys
import json

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Music Recommendation App 초기화 시작.")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    logging.debug(f"Added {current_dir} to sys.path for music_recommender.py")

from music_recommender import MusicRecommender, MockMusicRecommender

app = Flask(__name__)

GETSONGBPM_API_KEY = os.environ.get("GETSONGBPM_API_KEY")

if GETSONGBPM_API_KEY:
    logging.info("GETSONGBPM_API_KEY가 설정되어 실제 API를 사용 합니다.")
    recommender = MusicRecommender(GETSONGBPM_API_KEY)
else:
    logging.warning("GETSONGBPM_API_KEY 환경 변수가 설정되지 않았습니다. Mock 추천기를 사용합니다.")
    recommender = MockMusicRecommender()

logging.info(f"Recommender 객체 타입: {type(recommender).__name__}") # 추가된 로그
logging.info("음악 추천 모델 및 도구 로드 성공.")

@app.route('/')
def index():
    """
    메인 페이지를 렌더링합니다.
    """
    # 오류 메시지가 있다면 함께 렌더링
    error_message = request.args.get('error_message')
    return render_template('index.html', error_message=error_message)

@app.route('/recommend', methods=['POST'])
def recommend_music_endpoint():
    """
    사용자 메시지를 받아 음악을 추천하고 결과를 반환합니다.
    """
    user_text = request.form.get('user_message') # 'user_message'로 받음
    
    if not user_text or user_text.strip() == '':
        logging.warning("User message is empty. Returning 400 Bad Request.")
        # 클라이언트 JavaScript에서 이 JSON 응답을 받아서 처리할 것입니다.
        return jsonify({"error": "감정을 입력해주세요!"}), 400

    logging.info(f"\n--- Starting music recommendation for '{user_text}' ---")
    
    try:
        # 여기가 문제의 라인: recommender 객체에 recommend_music 메서드가 있어야 합니다.
        recommendation_info = recommender.recommend_music(user_text) 
        
        # 이제 Flask는 JSON 응답을 직접 보냅니다.
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
            recommendation_info = {"recommendations": [], "user_emotion": "알 수 없음", "target_audio_features": {"bpm": (0,0), "danceability": (0,0), "acousticness": (0,0)}}
    
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