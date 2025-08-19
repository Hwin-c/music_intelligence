# music_recommender.py (수정 완료)

import requests
import random
import os
import logging
import time
import json

# ======================================================================
# 1. 표준 상대 경로 임포트로 변경 (sys.path 조작 제거)
# ======================================================================
# '.'은 현재 패키지(music_recommendation_app)를 의미합니다.
# 이 코드는 __init__.py 파일 덕분에 정상적으로 작동합니다.
from .natural_language_processing.sentiment_analyzer import SentimentAnalyzer
from .natural_language_processing.bpm_mapper import BPMMapper
# ======================================================================

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 오디오 특성 기반 관련성 점수 계산 함수 (변경 없음) ---
def _calculate_relevance_score(song_data: dict, target_features: dict, weights: dict) -> float:
    score = 0.0
    def calculate_feature_score(song_value, min_target, max_target, is_tempo=False):
        if song_value is None:
            return 0.0
        try:
            if is_tempo and isinstance(song_value, float) and (0 <= song_value <= 1):
                song_value = int(song_value * 100)
            else:
                song_value = int(song_value)

            if min_target <= song_value <= max_target:
                center_target = (min_target + max_target) / 2
                range_half = (max_target - min_target) / 2
                if range_half > 0:
                    return (1 - abs(song_value - center_target) / range_half) * 0.5 + 0.5
                else:
                    return 1.0 if song_value == min_target else 0.0
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    score += calculate_feature_score(song_data.get("tempo"), *target_features["bpm"], is_tempo=True) * weights.get("bpm", 1.0)
    score += calculate_feature_score(song_data.get("danceability"), *target_features["danceability"]) * weights.get("danceability", 1.0)
    score += calculate_feature_score(song_data.get("acousticness"), *target_features["acousticness"]) * weights.get("acousticness", 1.0)
    return score

# --- 감정 카테고리 매핑 및 가중치 (변경 없음) ---
broad_emotion_category_map = {
    "긍정": "긍정", "기쁨": "긍정", "신이 난": "긍정", "흥분": "긍정", "감사하는": "긍정", "신뢰하는": "긍정", "만족스러운": "긍정", "자신하는": "긍정",
    "부정": "부정", "슬픔": "부정", "우울한": "부정", "비통한": "부정", "후회되는": "부정", "낙담한": "부정", "마비된": "부정", "염세적인": "부정", "눈물이 나는": "부정", "실망한": "부정", "환멸을 느끼는": "부정", "취약한": "부정", "상처": "부정", "질투하는": "부정", "배신당한": "부정", "고립된": "부정", "충격 받은": "부정", "가난한 불우한": "부정", "희생된": "부정", "억울한": "부정", "괴로워하는": "부정", "외로운": "부정", "열등감": "부정", "죄책감의": "부정", "부끄러운": "부정", "한심한": "부정", "혼란스러운": "부정", "당혹스러운": "부정", "회의적인": "부정", "조심스러운": "부정", "걱정스러운": "부정", "초조한": "부정", "불안": "부정", "고립된(당황한)": "부정", "남의 시선을 의식하는": "부정", "혼란스러운(당황한)": "부정",
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
}

class MockMusicRecommender(object):
    """
    API 키가 없거나 개발 시에 사용할 모의(Mock) 추천기입니다.
    실제 API 호출 없이 미리 정의된 데이터를 반환합니다.
    """
    def __init__(self, getsongbpm_api_key: str = None):
        # 이제 Mock Recommender도 실제 분석/매퍼 모듈을 사용합니다.
        # "Mock"의 의미는 "데이터 소스가 Mock"이라는 뜻으로 한정됩니다.
        self.sentiment_analyzer = SentimentAnalyzer()
        self.bpm_mapper = BPMMapper()
        logging.info("--- Mock Music Recommender initialized (using mock data only) ---")

    def get_ranked_songs_by_audio_features(self, emotion_label: str, limit: int = 3):
        logging.info("Mock API call: Simulating search and ranking for '%s' emotion...", emotion_label)
        time.sleep(1)
        
        target_audio_features = self.bpm_mapper.get_audio_feature_ranges(emotion_label)
        broad_category = broad_emotion_category_map.get(emotion_label, "긍정")
        current_weights = emotion_weights.get(broad_category, emotion_weights["긍정"])

        mock_songs_data = [
            {"title": "Dancing Monkey (Mock)", "artist": "Tones And I (Mock)", "bpm": 98, "uri": "#", "genres": ["Pop", "Indie"], "danceability": 80, "acousticness": 10},
            {"title": "Shape of You (Mock)", "artist": "Ed Sheeran (Mock)", "bpm": 96, "uri": "#", "genres": ["Pop", "R&B"], "danceability": 85, "acousticness": 5},
            {"title": "Someone You Loved (Mock)", "artist": "Lewis Capaldi (Mock)", "bpm": 109, "uri": "#", "genres": ["Pop", "Ballad"], "danceability": 30, "acousticness": 80},
            {"title": "Happy (Mock)", "artist": "Pharrell Williams (Mock)", "bpm": 160, "uri": "#", "genres": ["Pop", "Soul"], "danceability": 95, "acousticness": 5},
            {"title": "Imagine (Mock)", "artist": "John Lennon (Mock)", "bpm": 75, "uri": "#", "genres": ["Pop", "Soft Rock"], "danceability": 20, "acousticness": 90},
        ]
        
        for song in mock_songs_data:
            song["relevance_score"] = _calculate_relevance_score(
                {"tempo": song["bpm"], "danceability": song["danceability"], "acousticness": song["acousticness"]},
                target_audio_features,
                current_weights
            )
        
        mock_songs_data.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return mock_songs_data[:limit]

    def recommend_music(self, user_text: str, limit: int = 3):
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_text)
        user_emotion = sentiment_result["label"]
        target_audio_features = self.bpm_mapper.get_audio_feature_ranges(user_emotion)
        recommended_songs = self.get_ranked_songs_by_audio_features(user_emotion, limit=limit)
        return {
            "user_emotion": user_emotion,
            "target_audio_features": target_audio_features,
            "recommendations": recommended_songs
        }

class MusicRecommender:
    """
    사용자의 감정을 분석하여 음악 특성 기반으로 음악을 추천하는 클래스입니다.
    """
    def __init__(self, getsongbpm_api_key: str):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.bpm_mapper = BPMMapper()
        self.getsongbpm_api_key = getsongbpm_api_key
        self.getsongbpm_base_url = "https://api.getsong.co/"
        logging.info("Music recommendation system initialized with API Base URL: %s", self.getsongbpm_base_url)

    def _call_getsongbpm_api(self, endpoint: str, params: dict = None, delay_seconds: float = 1.5):
        if params is None:
            params = {}
        params["api_key"] = self.getsongbpm_api_key
        url = f"{self.getsongbpm_base_url}{endpoint.lstrip('/')}"
        
        try:
            logging.debug("Calling getsong.co API: %s with params %s", url, params)
            response = requests.get(url, params=params)
            response.raise_for_status()
            time.sleep(delay_seconds)
            return response.json()
        except requests.exceptions.RequestException as err:
            logging.error("API call failed: %s", err)
            return None

    def get_ranked_songs_by_audio_features(self, emotion_label: str, limit: int = 3):
        logging.info("Searching songs for '%s' emotion via getsong.co API...", emotion_label)
        target_features = self.bpm_mapper.get_audio_feature_ranges(emotion_label)
        broad_category = broad_emotion_category_map.get(emotion_label, "긍정")
        current_weights = emotion_weights.get(broad_category, emotion_weights["긍정"])
        candidate_songs = []
        seen_titles = set()
        common_titles_to_filter_lower = [
            "tune", "mix", "track", "edit", "version", "remix", "instrumental", "intro", "outro", "live", "acoustic",
            "radio", "original", "album", "single", "theme", "song", "beat", "music", "vocal", "pop", "rock", "jazz",
        ]
        emotion_keywords_for_search = {
            "긍정": ["happy", "upbeat", "energetic", "joyful", "party", "dancing"],
            "부정": ["sad", "melancholy", "downbeat", "gloomy", "heartbreak", "sorrow"],
            "분노": ["angry", "rage", "intense", "aggressive", "metal", "punk"],
            "스트레스": ["relax", "calm", "chill", "soothing", "meditation", "peaceful"],
            "평온": ["relax", "calm", "chill", "peaceful", "smooth", "mellow"],
        }
        genre_keywords = ["pop", "dance", "rock", "electronic", "jazz", "hip hop", "ballad", "k-pop"]
        keywords_for_query = emotion_keywords_for_search.get(broad_category, emotion_keywords_for_search["긍정"])
        search_queries = list(set(keywords_for_query + genre_keywords))
        random.shuffle(search_queries)

        for query in search_queries[:5]: # API 호출 횟수 제한
            params = {"type": "song", "lookup": query, "limit": 50}
            api_response = self._call_getsongbpm_api("search/", params)
            
            if api_response and isinstance(api_response.get("search"), list):
                for song_data in api_response["search"]:
                    title = song_data.get("title")
                    if not title or title in seen_titles: continue
                    if title.lower() in common_titles_to_filter_lower: continue
                    
                    artist_info = song_data.get("artist", {})
                    artist_name = artist_info.get("name", "Unknown Artist")
                    
                    try:
                        bpm = int(song_data.get("tempo", 0))
                        danceability = int(song_data.get("danceability", 0))
                        acousticness = int(song_data.get("acousticness", 0))
                    except (ValueError, TypeError):
                        continue

                    relevance_score = _calculate_relevance_score(
                        {"tempo": bpm, "danceability": danceability, "acousticness": acousticness},
                        target_features,
                        current_weights
                    )

                    if relevance_score > 0:
                        candidate_songs.append({
                            "title": title, "artist": artist_name, "bpm": bpm,
                            "uri": song_data.get("uri", "#"), "genres": artist_info.get("genres", []),
                            "danceability": danceability, "acousticness": acousticness,
                            "relevance_score": relevance_score
                        })
                        seen_titles.add(title)
            if len(candidate_songs) >= limit * 5: break
        
        candidate_songs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return candidate_songs[:limit]

    def recommend_music(self, user_text: str, limit: int = 3):
        logging.info("Starting recommendation for: '%s'", user_text)
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_text)
        user_emotion = sentiment_result["label"]
        target_audio_features = self.bpm_mapper.get_audio_feature_ranges(user_emotion)
        recommended_songs = self.get_ranked_songs_by_audio_features(user_emotion, limit=limit)
        return {
            "user_emotion": user_emotion,
            "target_audio_features": target_audio_features,
            "recommendations": recommended_songs
        }

if __name__ == "__main__":
    getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY")

    if not getsongbpm_api_key:
        logging.warning("\n[WARNING]: GETSONGBPM_API_KEY not set. Using mock data.")
        recommender = MockMusicRecommender()
    else:
        logging.info("GETSONGBPM_API_KEY is set. Using actual API calls.")
        recommender = MusicRecommender(getsongbpm_api_key)

    test_inputs = [
        "오늘은 정말 기분이 좋고 신나!",
        "요즘 너무 우울해서 슬픈 노래가 듣고 싶어.",
        "정말 화가 나서 아무것도 못 하겠어.",
    ]

    for user_text in test_inputs:
        try:
            result = recommender.recommend_music(user_text)
            logging.info("\nUser Emotion: %s", result['user_emotion'])
            logging.info("Recommended Songs:")
            for song in result['recommendations']:
                logging.info("  - %s by %s (Score: %.2f)", song['title'], song['artist'], song.get('relevance_score', 0))
        except Exception as e:
            logging.error("Test failed for '%s': %s", user_text, e)