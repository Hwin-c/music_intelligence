from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import logging
import traceback
import sys

# Flask 앱 인스턴스 생성 시 static_folder와 template_folder를 명시적으로 설정
app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Music Recommendation App 초기화 시작.")

# --- 음악 추천 기능 관련 모듈 로드 (지연 로딩을 위해 전역 변수로 선언) ---
recommender = None
getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY", "YOUR_GETSONGBPM_API_KEY_HERE")

def _load_recommendation_models():
    """
    음악 추천 모델과 관련 도구들을 지연 로드하는 내부 함수.
    """
    global recommender
    
    if recommender is not None: # 이미 로드되었다면 다시 로드하지 않음
        logging.debug("음악 추천 모델이 이미 로드되어 있습니다. 다시 로드하지 않습니다.")
        return

    logging.debug("음악 추천 모델 지연 로드 시작.")
    try:
        # music_recommender 모듈을 임포트합니다.
        from music_recommender import MusicRecommender, MockMusicRecommender 

        if getsongbpm_api_key == "YOUR_GETSONGBPM_API_KEY_HERE":
            logging.warning("[WARNING]: getsongbpm API 키가 설정되지 않았습니다. 모의(Mock) 데이터를 사용하여 테스트를 진행합니다.")
            recommender = MockMusicRecommender(getsongbpm_api_key) # MockMusicRecommender 사용
        else:
            recommender = MusicRecommender(getsongbpm_api_key) # MusicRecommender 사용
        
        logging.info("음악 추천 모델 및 도구 지연 로드 성공.")
    except ImportError as e:
        logging.error(f"필수 모듈 임포트 실패: {e}")
        logging.error("music_recommender.py 및 관련 NLP 모듈이 올바른 경로에 있는지 확인하십시오.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"음악 추천 모델 지연 로드 중 오류 발생: {e}")
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
    사용자 텍스트를 입력받아 음악을 추천하는 엔드포인트입니다.
    """
    _load_recommendation_models() # 모델 로드

    user_text = request.form.get('user_text')
    if not user_text:
        logging.warning("사용자 텍스트가 제공되지 않았습니다.")
        return jsonify({'error': 'No user text provided'}), 400

    try:
        recommended_songs = recommender.recommend_music(user_text)
        
        # 앨범 커버 URL이 없는 경우 빈 문자열로 유지 (HTML에서 처리)
        for song in recommended_songs:
            if 'album_cover_url' not in song or not song['album_cover_url']:
                song['album_cover_url'] = '' # 빈 문자열로 설정
        
        return jsonify({'user_message': user_text, 'recommendations': recommended_songs}), 200

    except Exception as e:
        logging.error(f'음악 추천 중 오류 발생: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'error': f'음악 추천 중 오류 발생: {str(e)}'}), 500

@app.route('/recommend_result')
def recommend_result():
    """음악 추천 결과 페이지 렌더링."""
    # JSON 문자열로 받은 데이터를 파싱
    user_message = request.args.get('user_message', '요청하신 내용을 바탕으로 추천했어요…!')
    recommendations_str = request.args.get('recommendations', '[]')
    
    try:
        import json
        recommendations = json.loads(recommendations_str)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode recommendations JSON: {recommendations_str}")
        recommendations = []

    # 순위 아이콘 경로 설정
    rank_icons = {
        1: url_for('static', filename='1st.png'),
        2: url_for('static', filename='2nd.png'),
        3: url_for('static', filename='3rd.png'),
    }

    # 각 추천곡에 순위 아이콘 경로 추가
    for i, song in enumerate(recommendations):
        song['rank_icon'] = rank_icons.get(i + 1, '') # 1, 2, 3위만 아이콘 적용

    return render_template('recommend_result.html', 
                           user_message=user_message, 
                           recommendations=recommendations)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001)) # 추천 앱은 5001 포트 사용
    logging.debug(f"Music Recommendation App 로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)