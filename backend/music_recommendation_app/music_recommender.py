import requests
import random
import os
import logging
import time

from natural_language_processing.sentiment_analyzer import SentimentAnalyzer
from natural_language_processing.bpm_mapper import BPMMapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _calculate_relevance_score(song_data: dict, target_features: dict, weights: dict, sentiment_score: float) -> float:
    audio_feature_score = 0.0
    
    def calculate_feature_score(song_value, min_target, max_target):
        if song_value is None: return 0.0
        try:
            song_value = int(song_value)
            if min_target <= song_value <= max_target:
                center_target = (min_target + max_target) / 2
                range_half = (max_target - min_target) / 2
                if range_half > 0:
                    return (1 - abs(song_value - center_target) / range_half) * 0.5 + 0.5
                return 1.0 if song_value == min_target else 0.0
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    audio_feature_score += calculate_feature_score(song_data.get("tempo"), *target_features["bpm"]) * weights.get("bpm", 1.0)
    audio_feature_score += calculate_feature_score(song_data.get("danceability"), *target_features["danceability"]) * weights.get("danceability", 1.0)
    audio_feature_score += calculate_feature_score(song_data.get("acousticness"), *target_features["acousticness"]) * weights.get("acousticness", 1.0)
    
    total_weight = sum(weights.values())
    if total_weight == 0: return 0.0

    normalized_audio_score = audio_feature_score / total_weight
    
    # ★★★ 핵심 수정: 최종 점수를 100점 만점으로 환산 ★★★
    # 기존 점수(0~1 사이)에 100을 곱하여 사용자에게 더 직관적인 점수를 제공합니다.
    final_score = (normalized_audio_score * sentiment_score) * 100
    
    return final_score

broad_emotion_category_map = {
    "기쁨": "긍정", "신이 난": "긍정", "흥분": "긍정", "감사하는": "긍정", "신뢰하는": "긍정", "만족스러운": "긍정", "자신하는": "긍정",
    "슬픔": "부정", "우울한": "부정", "비통한": "부정", "후회되는": "부정", "낙담한": "부정", "마비된": "부정", "염세적인": "부정", "눈물이 나는": "부정", "실망한": "부정", "환멸을 느끼는": "부정", "취약한": "부정", "상처": "부정", "배신당한": "부정", "고립된": "부정", "충격 받은": "부정", "가난한 불우한": "부정", "희생된": "부정", "억울한": "부정", "괴로워하는": "부정", "외로운": "부정", "열등감": "부정", "죄책감의": "부정", "부끄러운": "부정", "한심한": "부정",
    "혼란스러운": "불안", "당혹스러운": "불안", "회의적인": "불안", "조심스러운": "불안", "걱정스러운": "불안", "초조한": "불안", "불안": "불안", "질투하는": "불안", "고립된(당황한)": "불안", "남의 시선을 의식하는": "불안", "혼란스러운(당황한)": "불안",
    "편안한": "평온", "느긋": "평온", "안도": "평온",
    "분노": "분노", "툴툴대는": "분노", "좌절한": "분노", "짜증내는": "분노", "방어적인": "분노", "악의적인": "분노", "안달하는": "분노", "혐오스러운": "분노",
    "스트레스 받는": "스트레스",
}

emotion_weights = {
    "긍정": {"bpm": 1.0, "danceability": 1.0, "acousticness": 0.2},
    "부정": {"bpm": 0.8, "danceability": 0.3, "acousticness": 1.0},
    "평온": {"bpm": 0.7, "danceability": 0.4, "acousticness": 0.9},
    "분노": {"bpm": 1.0, "danceability": 0.7, "acousticness": 0.1},
    "스트레스": {"bpm": 0.7, "danceability": 0.4, "acousticness": 0.9},
    "불안": {"bpm": 0.7, "danceability": 0.4, "acousticness": 0.9},
}

class MusicRecommender:
    def __init__(self, getsongbpm_api_key: str):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.bpm_mapper = BPMMapper()
        self.getsongbpm_api_key = getsongbpm_api_key
        self.getsongbpm_base_url = "https://api.getsong.co/"
        logging.info("음악 추천 시스템이 초기화되었습니다.")

    def _call_getsongbpm_api(self, endpoint: str, params: dict = None):
        if not self.getsongbpm_api_key:
            logging.warning("Getsong API 키가 없어 API를 호출할 수 없습니다.")
            return None
        
        params = params or {}
        params["api_key"] = self.getsongbpm_api_key
        url = f"{self.getsongbpm_base_url}{endpoint.lstrip('/')}"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            response_text = response.text
            if not response_text.strip():
                logging.warning(f"API 응답이 비어있습니다 (Query: {params.get('lookup')}).")
                return None
            
            return response.json()
        except requests.exceptions.JSONDecodeError:
            logging.error(f"Getsong API 응답 JSON 파싱 실패 (Query: {params.get('lookup')}). 수신된 텍스트: '{response.text[:200]}...'")
            return None
        except requests.exceptions.RequestException as err:
            logging.error(f"Getsong API 호출 중 오류 발생: {err}")
            return None

    def get_ranked_songs_by_audio_features(self, emotion_label: str, sentiment_score: float, limit: int = 3):
        logging.info(f"'{emotion_label}' 감정(점수: {sentiment_score:.2f})에 맞는 노래를 검색합니다...")
        target_features = self.bpm_mapper.get_audio_feature_ranges(emotion_label)
        broad_category = broad_emotion_category_map.get(emotion_label, "긍정")
        current_weights = emotion_weights.get(broad_category, emotion_weights["긍정"])
        emotion_keywords_for_search = {
            "긍정": ["happy", "upbeat", "energetic", "party", "dancing"],
            "부정": ["sad", "melancholy", "downbeat", "gloomy", "heartbreak", "chill"],
            "분노": ["angry", "rage", "intense", "aggressive", "metal", "punk"],
            "스트레스": ["relax", "calm", "chill", "soothing", "meditation"],
            "평온": ["relax", "calm", "chill", "peaceful", "smooth", "mellow"],
            "불안": ["soothing", "calm", "meditation", "peaceful", "gentle"],
        }
        keywords_for_query = emotion_keywords_for_search.get(broad_category, emotion_keywords_for_search["긍정"])
        
        candidate_songs = []
        seen_titles = set()
        
        random.shuffle(keywords_for_query)

        for query in keywords_for_query[:3]:
            if len(candidate_songs) >= limit: break

            logging.info(f"API 호출 시도: 검색 키워드 = '{query}'")
            params = {"type": "song", "lookup": query, "limit": 50}
            api_response = self._call_getsongbpm_api("search/", params)
            
            if api_response and isinstance(api_response.get("search"), list):
                for song_data in api_response["search"]:
                    title = song_data.get("title")
                    if not title or title.lower() in seen_titles: continue
                    
                    artist_name = song_data.get("artist", {}).get("name", "Unknown Artist")
                    
                    tempo = song_data.get("tempo")
                    danceability = song_data.get("danceability")
                    acousticness = song_data.get("acousticness")

                    if tempo is not None and danceability is not None and acousticness is not None:
                        try:
                            relevance_score = _calculate_relevance_score(song_data, target_features, current_weights, sentiment_score)

                            candidate_songs.append({
                                "title": title, "artist": artist_name,
                                "bpm": int(tempo),
                                "uri": song_data.get("uri", "#"),
                                "genres": song_data.get("artist", {}).get("genres", []),
                                "danceability": int(danceability),
                                "acousticness": int(acousticness),
                                "relevance_score": relevance_score
                            })
                            seen_titles.add(title.lower())
                        except (ValueError, TypeError) as e:
                            logging.warning(f"'{title}' 노래의 오디오 특성 값 변환 중 오류: {e}. 이 노래를 건너뜁니다.")
                            continue
                    else:
                        logging.warning(f"'{title}' 노래에 필수 오디오 특성이 누락되었습니다. 이 노래를 건너뜁니다.")
                        continue

            time.sleep(1.5)

        if not candidate_songs:
            logging.warning("모든 API 호출 시도 후에도 추천할 후보곡을 찾지 못했습니다.")

        candidate_songs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return candidate_songs[:limit]

    def recommend_music(self, user_text: str, limit: int = 3):
        logging.info(f"추천 프로세스 시작: '{user_text}'")
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_text)
        user_emotion = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]
        target_audio_features = self.bpm_mapper.get_audio_feature_ranges(user_emotion)
        recommended_songs = self.get_ranked_songs_by_audio_features(user_emotion, sentiment_score, limit=limit)
        return {
            "user_emotion": user_emotion,
            "target_audio_features": target_audio_features,
            "recommendations": recommended_songs
        }

# 테스트 코드
if __name__ == "__main__":
    getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY")

    if not getsongbpm_api_key:
        logging.warning("\n[테스트 경고]: GETSONGBPM_API_KEY 환경 변수가 설정되지 않았습니다.")
    else:
        recommender = MusicRecommender(getsongbpm_api_key)
        logging.info("\n==== 음악 추천 시스템 데모 시작 ====")
        test_inputs = [
            "오늘은 정말 기분이 좋고 신나는 하루였어!",
            "너무 우울해서 아무것도 하기 싫다.",
            "정말 화가 나서 미칠 것 같아."
        ]
        for user_text in test_inputs:
            result = recommender.recommend_music(user_text)
            logging.info(f"\n>> 사용자 입력: '{user_text}'")
            logging.info(f"   분석된 감정: {result['user_emotion']}")
            logging.info("   추천된 노래:")
            for song in result['recommendations']:
                logging.info(f"     - {song['title']} by {song['artist']} (최종 점수: {song.get('relevance_score', 0):.4f})")