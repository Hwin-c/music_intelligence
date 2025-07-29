from flask import Flask, render_template, redirect, url_for
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("통합 메인 앱 초기화 시작.")

# Flask 앱 인스턴스 생성
# templates 폴더는 이 app.py와 같은 레벨에 있는 'templates' 폴더를 참조합니다.
# static 폴더는 이 app.py와 같은 레벨에 있는 'static' 폴더를 참조합니다.
app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# 루트 경로 ('/')는 메인 랜딩 페이지를 렌더링합니다.
@app.route('/')
def main_landing():
    logging.info("메인 랜딩 페이지 요청 수신.")
    return render_template('main_landing_page.html')

# 음악 장르 분류 앱으로 리다이렉트하는 라우트
@app.route('/genre')
def redirect_to_genre_app():
    logging.info("음악 장르 분류 앱으로 리다이렉트.")
    # 실제 배포 환경에서는 여기에 music_genre_app의 실제 URL을 넣어야 합니다.
    # 예: return redirect("https://your-genre-app.onrender.com")
    # 로컬 테스트를 위해선 Flask 개발 서버의 기본 포트(5000)를 사용한다고 가정합니다.
    # 만약 music_genre_app이 별도의 포트에서 실행된다면 해당 포트를 명시해야 합니다.
    # 현재는 로컬에서 하나의 Flask 앱으로 통합하는 것이 아니므로,
    # 각 앱이 별도의 프로세스로 실행되고 있다고 가정하고,
    # 장르 앱의 루트 경로로 리다이렉트합니다.
    # 이 부분은 배포 환경에 따라 달라질 수 있습니다.
    # 지금은 로컬에서 테스트하기 위해 임시로 /genre로 설정합니다.
    # 실제 배포 시에는 'http://127.0.0.1:5000' 대신 실제 도메인을 사용해야 합니다.
    return redirect("http://127.0.0.1:5000") # music_genre_app의 기본 URL (예시)

# 음악 추천 앱으로 리다이렉트하는 라우트
@app.route('/recommendation')
def redirect_to_recommendation_app():
    logging.info("음악 추천 앱으로 리다이렉트.")
    # 실제 배포 환경에서는 여기에 music_recommendation_app의 실제 URL을 넣어야 합니다.
    # 예: return redirect("https://your-recommendation-app.onrender.com")
    # 로컬 테스트를 위해선 Flask 개발 서버의 기본 포트(5001 등)를 사용한다고 가정합니다.
    # 만약 music_recommendation_app이 별도의 포트에서 실행된다면 해당 포트를 명시해야 합니다.
    # 지금은 로컬에서 테스트하기 위해 임시로 /recommendation으로 설정합니다.
    # 실제 배포 시에는 'http://127.0.0.1:5001' 대신 실제 도메인을 사용해야 합니다.
    return redirect("http://127.0.0.1:5001") # music_recommendation_app의 기본 URL (예시)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000)) # 메인 앱은 8000번 포트 사용 (충돌 방지)
    logging.info(f"통합 메인 앱 로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    # debug=True는 개발용이며, 프로덕션에서는 False로 설정해야 합니다.
    app.run(debug=True, host='0.0.0.0', port=port)