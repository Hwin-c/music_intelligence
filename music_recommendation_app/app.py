# C:\Users\USER\Desktop\music_intelligence\music_recommendation_app\app.py

from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import logging
import traceback
import sys
import threading
import time
import json
import requests

# 로깅 설정 (기존과 동일)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# --- Getsong API 연동을 위한 클래스 ---
class MusicRecommender:
    def __init__(self):
        # TODO: Getsong API 키 환경 변수 이름을 GETSONGBPM_API_KEY로 통일
        self.api_key = os.environ.get("GETSONGBPM_API_KEY", "YOUR_GETSONG_API_KEY_PLACEHOLDER") 
        # TODO: Getsong API의 텍스트 기반 추천 엔드포인트 실제 URL로 교체
        # 이 URL이 404 오류의 주 원인일 가능성이 높습니다. Getsong API 문서를 다시 확인하세요.
        self.api_url = os.environ.get("GETSONG_API_URL", "https://api.getsong.co/v1/recommend_by_text_actual") # 가상 URL, 실제 URL로 교체 필요
        
        if self.api_key == "YOUR_GETSONG_API_KEY_PLACEHOLDER":
            logging.warning("경고: Getsong API 키가 기본값으로 설정되어 있습니다. 실제 키로 교체해주세요.")
        if self.api_url == "https://api.getsong.co/v1/recommend_by_text_actual":
            logging.warning("경고: Getsong API URL이 가상값으로 설정되어 있습니다. 실제 URL로 교체해주세요.")

        logging.info("MusicRecommender 초기화 완료.")

    def recommend_music(self, user_message):
        logging.info(f"음악 추천 요청 수신: '{user_message}'")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Getsong API의 요청 바디 형식에 맞게 조정해야 합니다.
        # 예시: 텍스트 메시지를 'query' 필드에 담아 보낸다고 가정
        payload = {
            "query": user_message
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30) # 타임아웃 30초 설정
            response.raise_for_status() # HTTP 오류가 발생하면 예외 발생

            api_data = response.json()
            logging.info(f"Getsong API 응답 수신: {api_data}")

            # Getsong API의 실제 응답 구조에 따라 데이터를 파싱하고 변환해야 합니다.
            # 여기서는 이전 Mock 데이터 구조와 유사하게 매핑하는 예시입니다.
            # 실제 API 응답에 'emotion', 'target_features', 'tracks' 등의 필드가 있다고 가정합니다.
            
            # API 응답에서 감정, 특성, 추천곡 추출 (API 응답 구조에 따라 수정 필요)
            user_emotion = api_data.get("emotion", "분석 불가")
            target_audio_features = api_data.get("target_features", {
                "bpm": [90, 130], "danceability": [40, 70], "acousticness": [30, 70]
            })
            raw_recommendations = api_data.get("tracks", [])

            recommendations = []
            for track in raw_recommendations:
                # API 응답 필드명을 HTML에서 사용하는 필드명으로 매핑
                recommendations.append({
                    "title": track.get("title", "알 수 없는 제목"),
                    "artist": track.get("artist", "알 수 없는 아티스트"),
                    "uri": track.get("external_urls", {}).get("spotify") or track.get("youtube_url", "#"), # Spotify URL 또는 Youtube URL
                    "genres": track.get("genres", []),
                    "bpm": track.get("tempo", 0), # Getsong API에서 'tempo'로 제공될 수 있음
                    "danceability": track.get("danceability", 0) * 100, # 0-1 값을 0-100으로 변환
                    "acousticness": track.get("acousticness", 0) * 100, # 0-1 값을 0-100으로 변환
                    "relevance_score": track.get("score", 0) # API에서 제공하는 적합성 점수
                })
            
            # API에서 점수 정렬이 안 되어 있다면 여기서 다시 정렬
            recommendations.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            return {
                "user_emotion": user_emotion,
                "target_audio_features": target_audio_features,
                "recommendations": recommendations
            }

        except requests.exceptions.Timeout:
            logging.error("Getsong API 요청 시간 초과.")
            raise Exception("Getsong API 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Getsong API 요청 중 네트워크 또는 HTTP 오류 발생: {e}")
            # 이 부분에서 404 오류가 발생했으므로, 사용자에게 더 명확한 메시지를 전달합니다.
            if "404 Client Error" in str(e):
                raise Exception(f"Getsong API 엔드포인트 URL이 잘못되었거나 존재하지 않습니다: {self.api_url}")
            else:
                raise Exception(f"Getsong API 통신 중 오류가 발생했습니다: {str(e)}")
        except json.JSONDecodeError:
            logging.error(f"Getsong API 응답 JSON 파싱 실패: {response.text}")
            raise Exception("Getsong API 응답 형식이 올바르지 않습니다.")
        except Exception as e:
            logging.error(f"Getsong API 응답 처리 중 알 수 없는 오류 발생: {e}")
            raise Exception(f"음악 추천 데이터를 처리하는 중 오류가 발생했습니다: {str(e)}")


# MusicRecommender 인스턴스 생성
recommender = MusicRecommender()

# Flask 라우트 정의 (기존과 동일)
@app.route('/')
@app.route('/recommendation')
def recommendation_index():
    logging.info("음악 추천 앱 메인 페이지 요청 수신.")
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_music_endpoint():
    try:
        user_message = request.form.get('user_message')
        if not user_message:
            return jsonify({"error": "사용자 메시지가 제공되지 않았습니다."}), 400

        logging.info(f"--- Starting music recommendation for '{user_message}' ---")

        # MusicRecommender의 recommend_music 메서드 호출 (실제 API 호출)
        recommendation_info = recommender.recommend_music(user_message)
        
        # 클라이언트로 보낼 데이터 구성
        response_data = {
            "user_message": user_message,
            "recommendation_info": recommendation_info
        }
        return jsonify(response_data), 200

    except Exception as e:
        logging.error(f"음악 추천 중 오류 발생: {e}")
        traceback.print_exc(file=sys.stdout) # 콘솔에 전체 트레이스백 출력
        return jsonify({"error": f"음악 추천 처리 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/recommend_result')
def recommend_result_page():
    user_message = request.args.get('user_message')
    recommendation_info_json = request.args.get('recommendation_info_json')
    
    recommendation_info = {}
    if recommendation_info_json:
        try:
            recommendation_info = json.loads(recommendation_info_json)
        except json.JSONDecodeError:
            logging.error("recommendation_info_json 파싱 오류.")
            recommendation_info = {"error": "추천 정보를 불러오는 데 실패했습니다."}

    return render_template('recommend_result_page.html', 
                           user_message=user_message, 
                           recommendation_info=recommendation_info)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    logging.info(f"음악 추천 앱 로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)