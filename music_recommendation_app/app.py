from flask import Flask, request, jsonify
import os
import logging
import traceback
import sys

# music_recommendation_app 디렉토리를 Python 경로에 추가하여 내부 모듈 임포트 가능하도록 설정
# app.py가 music_recommendation_app/app.py 이므로, 현재 디렉토리를 sys.path에 추가
project_root = os.path.dirname(os.path.abspath(__file__))

logging.debug(f"Music Recommendation App: 프로젝트 루트: {project_root}")

if project_root not in sys.path:
    sys.path.append(project_root)
    logging.debug(f"Music Recommendation App: '{project_root}'를 sys.path에 추가했습니다.")

try:
    logging.debug("Music Recommendation App: music_recommender 모듈 임포트 시도 중...")
    # music_recommender.py가 music_recommendation_app 바로 아래에 있으므로 직접 임포트
    from music_recommender import MusicRecommender
    logging.debug("Music Recommendation App: music_recommender 모듈 임포트 성공.")
except ImportError as e:
    logging.error(f"Music Recommendation App: 오류: 'music_recommender' 모듈을 임포트할 수 없습니다. {e}")
    logging.error(f"Music Recommendation App: {traceback.format_exc()}")
    sys.exit(1)

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Music Recommendation App 초기화 시작.")

# --- 음악 추천 기능 관련 MusicRecommender 인스턴스 초기화 ---
# MusicRecommender는 자체적으로 SentimentAnalyzer를 지연 로드하므로, 여기서는 변경 없음
getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY", "YOUR_GETSONGBPM_API_KEY_HERE")

logging.debug(f"Music Recommendation App: MusicRecommender 인스턴스 초기화 시도 중 (API Key 존재 여부: {getsongbpm_api_key != 'YOUR_GETSONGBPM_API_KEY_HERE'})...")
try:
    recommender = MusicRecommender(getsongbpm_api_key)
    logging.info("Music Recommendation App: 음악 추천 시스템 인스턴스 초기화 완료.")
except Exception as e:
    logging.error(f"Music Recommendation App: 음악 추천 시스템 인스턴스 초기화 중 오류 발생: {e}")
    logging.error(traceback.format_exc())
    sys.exit(1) # MusicRecommender 초기화 실패 시 앱 종료

@app.route('/healthz')
def health_check():
    """Render 헬스 체크를 위한 엔드포인트."""
    logging.debug("Health check 요청 수신.")
    return "OK", 200

@app.route('/recommend', methods=['POST'])
def recommend_music_endpoint():
    """
    사용자 텍스트를 입력받아 감정 기반 음악을 추천하는 엔드포인트입니다.
    """
    logging.debug("Music Recommendation App: 음악 추천 요청 수신.")
    data = request.get_json()
    user_text = data.get('text')

    if not user_text:
        logging.warning("Music Recommendation App: 추천 요청에 텍스트가 없습니다.")
        return jsonify({'error': '텍스트를 입력해주세요.'}), 400

    try:
        recommended_songs = recommender.recommend_music(user_text)

        if recommended_songs:
            logging.info(f"Music Recommendation App: '{user_text}'에 대한 음악 추천 완료. {len(recommended_songs)}곡 추천.")
            return jsonify({'recommendations': recommended_songs}), 200
        else:
            logging.info(f"Music Recommendation App: '{user_text}'에 대한 추천 음악을 찾을 수 없습니다.")
            return jsonify({'message': '추천 음악을 찾을 수 없습니다.'}), 200

    except Exception as e:
        logging.error(f"Music Recommendation App: 음악 추천 중 오류 발생: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'음악 추천 중 오류 발생: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001)) # 장르 앱과 다른 포트 사용 (로컬 테스트용)
    logging.debug(f"Music Recommendation App 로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)