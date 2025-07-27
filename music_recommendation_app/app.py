from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import logging
import traceback
import sys
import json # JSON.parse를 위해 추가

# music_recommendation_app 디렉토리를 Python 경로에 추가하여 내부 모듈 임포트 가능하도록 설정
project_root = os.path.dirname(os.path.abspath(__file__))

logging.debug(f"Music Recommendation App: 프로젝트 루트: {project_root}")

if project_root not in sys.path:
    sys.path.append(project_root)
    logging.debug(f"Music Recommendation App: '{project_root}'를 sys.path에 추가했습니다.")

try:
    logging.debug("Music Recommendation App: music_recommender 모듈 임포트 시도 중...")
    from music_recommender import MusicRecommender
    logging.debug("Music Recommendation App: music_recommender 모듈 임포트 성공.")
except ImportError as e:
    logging.error(f"Music Recommendation App: 오류: 'music_recommender' 모듈을 임포트할 수 없습니다. {e}")
    logging.error(f"Music Recommendation App: {traceback.format_exc()}")
    sys.exit(1)

app = Flask(__name__)
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates') # templates 경로 설정

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Music Recommendation App 초기화 시작.")

# --- 음악 추천 기능 관련 MusicRecommender 인스턴스 초기화 ---
getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY", "YOUR_GETSONGBPM_API_KEY_HERE")

logging.debug(f"Music Recommendation App: MusicRecommender 인스턴스 초기화 시도 중 (API Key 존재 여부: {getsongbpm_api_key != 'YOUR_GETSONGBPM_API_KEY_HERE'})...")
try:
    recommender = MusicRecommender(getsongbpm_api_key)
    logging.info("Music Recommendation App: 음악 추천 시스템 인스턴스 초기화 완료.")
except Exception as e:
    logging.error(f"Music Recommendation App: 음악 추천 시스템 인스턴스 초기화 중 오류 발생: {e}")
    logging.error(traceback.format_exc())
    sys.exit(1)

@app.route('/')
def index():
    return render_template('index.html')

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

        # 추천 결과를 JSON 문자열로 직렬화하여 URL 파라미터로 전달
        # 추천 결과가 복잡할 경우, 세션 또는 데이터베이스를 사용하는 것이 더 좋지만,
        # 간단한 데모를 위해 URL 파라미터를 사용합니다.
        recommended_songs_json = json.dumps(recommended_songs)
        
        # 결과 페이지로 리다이렉트
        return redirect(url_for('show_recommend_result', 
                                query=user_text, 
                                recommendations=recommended_songs_json))

    except Exception as e:
        logging.error(f"Music Recommendation App: 음악 추천 중 오류 발생: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'음악 추천 중 오류 발생: {str(e)}'}), 500

@app.route('/result')
def show_recommend_result():
    """음악 추천 결과 페이지 렌더링."""
    user_query = request.args.get('query', '알 수 없음')
    recommendations_json = request.args.get('recommendations', '[]')
    
    # URL 파라미터로 받은 JSON 문자열을 파싱하여 템플릿으로 전달
    try:
        recommendations = json.loads(recommendations_json)
    except json.JSONDecodeError:
        recommendations = []
        logging.error("추천 데이터를 파싱하는 데 실패했습니다.")

    return render_template('recommend_result_page.html', 
                           user_query=user_query, 
                           recommendations=recommendations)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001)) # 장르 앱과 다른 포트 사용 (로컬 테스트용)
    logging.debug(f"Music Recommendation App 로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)