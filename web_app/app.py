from flask import Flask, request, jsonify, render_template
import os
import logging
import requests  # 워커 서비스와 통신하기 위해 추가
import traceback
import sys

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한 설정

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Web App (API Gateway) 초기화 시작.")

# Render Private Network에서 워커 서비스의 내부 URL
# 이 값들은 Render 대시보드에서 워커 서비스의 이름을 설정할 때 사용한 이름과 일치해야 합니다.
# 예: Render 서비스 이름이 'music-genre-worker'이면 URL은 'http://music-genre-worker:10000'
GENRE_WORKER_URL = os.environ.get("GENRE_WORKER_URL", "http://localhost:10001")  # 로컬 테스트용 기본값
RECOMMEND_WORKER_URL = os.environ.get("RECOMMEND_WORKER_URL", "http://localhost:10002")  # 로컬 테스트용 기본값

logging.info(f"장르 워커 URL: {GENRE_WORKER_URL}")
logging.info(f"추천 워커 URL: {RECOMMEND_WORKER_URL}")

@app.route('/')
def index():
    """메인 인덱스 페이지 렌더링."""
    logging.debug("메인 페이지 요청 수신.")
    return render_template('index.html')

@app.route('/healthz')
def health_check():
    """Render 헬스 체크를 위한 엔드포인트."""
    logging.debug("Health check 요청 수신.")
    return "OK", 200

@app.route('/predict', methods=['POST'])
def predict_genre():
    """
    음악 장르 분류 요청을 genre_worker로 전달합니다.
    """
    logging.debug("장르 분류 요청 수신. 워커로 전달.")
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['audio']
    files = {'audio': (audio_file.filename, audio_file.stream, audio_file.content_type)}

    try:
        # genre_worker로 요청 전달
        response = requests.post(f"{GENRE_WORKER_URL}/predict_genre", files=files, timeout=120)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

        logging.info(f"장르 워커 응답 수신: {response.status_code}")
        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        logging.error("장르 워커 응답 시간 초과.")
        return jsonify({'error': '장르 분류 서비스 응답 시간 초과'}), 504
    except requests.exceptions.ConnectionError as e:
        logging.error(f"장르 워커 연결 오류: {e}")
        return jsonify({'error': f'장르 분류 서비스에 연결할 수 없습니다: {e}'}), 503
    except Exception as e:
        logging.error(f"장르 분류 요청 처리 중 오류 발생: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'장르 분류 중 오류 발생: {str(e)}'}), 500

@app.route('/recommend', methods=['POST'])
def recommend_music_endpoint():
    """
    음악 추천 요청을 recommend_worker로 전달합니다.
    """
    logging.debug("음악 추천 요청 수신. 워커로 전달.")
    data = request.get_json()
    user_text = data.get('text')

    if not user_text:
        return jsonify({'error': '텍스트를 입력해주세요.'}), 400

    try:
        # recommend_worker로 요청 전달
        response = requests.post(f"{RECOMMEND_WORKER_URL}/recommend_music", json={'text': user_text}, timeout=120)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

        logging.info(f"추천 워커 응답 수신: {response.status_code}")
        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        logging.error("추천 워커 응답 시간 초과.")
        return jsonify({'error': '음악 추천 서비스 응답 시간 초과'}), 504
    except requests.exceptions.ConnectionError as e:
        logging.error(f"추천 워커 연결 오류: {e}")
        return jsonify({'error': f'음악 추천 서비스에 연결할 수 없습니다: {e}'}), 503
    except Exception as e:
        logging.error(f"음악 추천 요청 처리 중 오류 발생: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'음악 추천 중 오류 발생: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.debug(f"로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)