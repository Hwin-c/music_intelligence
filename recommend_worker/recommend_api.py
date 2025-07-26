from flask import Flask, request, jsonify
import os
import logging
import traceback
import sys

# 로깅 설정 (먼저 설정)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Recommend Worker 초기화 시작.")

# 프로젝트 경로 설정
project_root = os.path.dirname(os.path.abspath(__file__))
music_rec_dir = os.path.join(project_root, 'music-recommendation-by-bpm')

logging.debug(f"음악 추천 워커: 프로젝트 루트: {project_root}")
logging.debug(f"음악 추천 워커: 음악 추천 모듈 디렉토리: {music_rec_dir}")

if music_rec_dir not in sys.path:
    sys.path.append(music_rec_dir)
    logging.debug(f"음악 추천 워커: '{music_rec_dir}'를 sys.path에 추가했습니다.")

# 모듈 임포트
try:
    logging.debug("음악 추천 워커: music_recommender 모듈 임포트 시도 중...")
    from music_recommender import MusicRecommender
    logging.debug("음악 추천 워커: music_recommender 모듈 임포트 성공.")
except ImportError as e:
    logging.error(f"음악 추천 워커: 오류: 'music_recommender' 모듈을 임포트할 수 없습니다. {e}")
    logging.error(traceback.format_exc())
    sys.exit(1)

# Flask 앱 초기화
app = Flask(__name__)

# MusicRecommender 인스턴스 초기화
getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY", "YOUR_GETSONGBPM_API_KEY_HERE")
logging.debug(f"Recommend Worker: MusicRecommender 인스턴스 초기화 시도 중 (API Key 존재 여부: {getsongbpm_api_key != 'YOUR_GETSONGBPM_API_KEY_HERE'})...")

try:
    recommender = MusicRecommender(getsongbpm_api_key)
    logging.info("Recommend Worker: 음악 추천 시스템 인스턴스 초기화 완료.")
except Exception as e:
    logging.error(f"Recommend Worker: 음악 추천 시스템 초기화 중 오류 발생: {e}")
    logging.error(traceback.format_exc())
    sys.exit(1)

# Health check 엔드포인트
@app.route('/healthz')
def health_check():
    logging.debug("Recommend Worker Health check 요청 수신.")
    return "OK", 200

# 음악 추천 엔드포인트
@app.route('/recommend_music', methods=['POST'])
def recommend_music_endpoint():
    logging.debug("Recommend Worker: 음악 추천 요청 수신.")
    data = request.get_json()
    user_text = data.get('text')

    if not user_text:
        logging.warning("Recommend Worker: 추천 요청에 텍스트가 없습니다.")
        return jsonify({'error': '텍스트를 입력해주세요.'}), 400

    try:
        recommended_songs = recommender.recommend_music(user_text)

        if recommended_songs:
            logging.info(f"'{user_text}'에 대한 음악 추천 완료. {len(recommended_songs)}곡 추천.")
            return jsonify({'recommendations': recommended_songs}), 200
        else:
            logging.info(f"'{user_text}'에 대한 추천 음악을 찾을 수 없습니다.")
            return jsonify({'message': '추천 음악을 찾을 수 없습니다.'}), 200

    except Exception as e:
        logging.error(f"Recommend Worker: 음악 추천 중 오류 발생: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'음악 추천 중 오류 발생: {str(e)}'}), 500

# 애플리케이션 실행
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10002))  # 워커는 별도 포트 사용
    logging.debug(f"Recommend Worker 로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)