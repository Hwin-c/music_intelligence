from flask import Flask, request, jsonify # render_template는 이제 필요 없음
import os
import logging
import traceback
import sys

# --- 음악 추천 기능 관련 모듈 임포트 ---
# music-recommendation-by-bpm 디렉토리를 Python 경로에 추가하여 패키지 임포트가 가능하도록 합니다.
project_root = os.path.dirname(os.path.abspath(__file__))
music_rec_dir = os.path.join(project_root, 'music-recommendation-by-bpm')

logging.debug(f"프로젝트 루트: {project_root}")
logging.debug(f"음악 추천 모듈 디렉토리: {music_rec_dir}")

if music_rec_dir not in sys.path:
    sys.path.append(music_rec_dir)
    logging.debug(f"'{music_rec_dir}'를 sys.path에 추가했습니다.")

try:
    logging.debug("music_recommender 모듈 임포트 시도 중...")
    from music_recommender import MusicRecommender
    logging.debug("music_recommender 모듈 임포트 성공.")
except ImportError as e:
    logging.error(f"오류: 'music_recommender' 모듈을 임포트할 수 없습니다. {e}")
    logging.error("다음 사항을 확인해주세요:")
    logging.error("1. 'music-recommendation-by-bpm' 디렉토리 안에 '__init__.py' 파일이 있는지 확인.")
    logging.error("2. 'music_recommender.py' 파일이 'music-recommendation-by-bpm' 디렉토리 안에 있는지 확인.")
    logging.error("3. 가상 환경이 활성화되어 있고 필요한 라이브러리(transformers, torch, requests)가 설치되었는지 확인.")
    sys.exit(1) # 모듈 로드 실패 시 애플리케이션 종료

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한 설정

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Flask 앱 초기화 시작.")

# --- 음악 장르 분류 기능 관련 모델 및 도구 로드 부분 삭제 ---
# 이 부분은 이제 필요 없습니다.

# --- 2. 음악 추천 기능 관련 MusicRecommender 인스턴스 초기화 ---
getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY", "YOUR_GETSONGBPM_API_KEY_HERE")

logging.debug(f"MusicRecommender 인스턴스 초기화 시도 중 (API Key 존재 여부: {getsongbpm_api_key != 'YOUR_GETSONGBPM_API_KEY_HERE'})...")
try:
    recommender = MusicRecommender(getsongbpm_api_key)
    logging.info("음악 추천 시스템 인스턴스 초기화 완료.")
except Exception as e:
    logging.error(f"음악 추천 시스템 인스턴스 초기화 중 오류 발생: {e}")
    logging.error(traceback.format_exc())
    sys.exit(1) # MusicRecommender 초기화 실패 시 앱 종료

# --- Render 헬스 체크 엔드포인트 ---
@app.route('/healthz')
def health_check():
    """
    Render 헬스 체크를 위한 엔드포인트.
    앱이 성공적으로 로드되면 200 OK를 반환합니다.
    """
    logging.debug("Health check 요청 수신.")
    return "OK", 200

# --- 음악 장르 분류 기능 엔드포인트 삭제 ---
# @app.route('/') def index(): ...
# @app.route('/predict', methods=['POST']) def predict(): ...
# extract_features_from_audio 함수도 삭제됩니다.

# --- 2. 음악 추천 기능 엔드포인트 ---
@app.route('/recommend', methods=['POST'])
def recommend_music_endpoint():
    """
    사용자 텍스트를 입력받아 감정 기반 음악을 추천하는 엔드포인트입니다.
    """
    data = request.get_json()
    user_text = data.get('text')

    if not user_text:
        logging.warning("추천 요청에 텍스트가 없습니다.")
        return jsonify({'error': '텍스트를 입력해주세요.'}), 400

    try:
        # MusicRecommender의 recommend_music 메서드 호출
        recommended_songs = recommender.recommend_music(user_text)

        # 웹 응답을 위해 추천된 노래 리스트만 JSON으로 반환합니다.
        if recommended_songs:
            logging.info(f"'{user_text}'에 대한 음악 추천 완료. {len(recommended_songs)}곡 추천.")
            return jsonify({'recommendations': recommended_songs}), 200
        else:
            logging.info(f"'{user_text}'에 대한 추천 음악을 찾을 수 없습니다.")
            return jsonify({'message': '추천 음악을 찾을 수 없습니다.'}), 200

    except Exception as e:
        logging.error(f"음악 추천 중 오류 발생: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'음악 추천 중 오류 발생: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.debug(f"로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)