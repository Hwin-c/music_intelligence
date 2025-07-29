import requests
import random
import os
import sys
import logging
import time
import json

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# natural-language-processing 디렉토리를 Python 경로에 추가하여 모듈을 임포트할 수 있도록 합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
nlp_dir = os.path.join(current_dir, 'natural-language-processing')

if nlp_dir not in sys.path:
    sys.path.insert(0, nlp_dir)
    logging.debug(f"Added {nlp_dir} to sys.path for NLP modules.")

# --- Mock SentimentAnalyzer 및 BPMMapper 클래스들을 먼저 정의합니다. ---
class SentimentAnalyzer:
    def analyze_sentiment(self, text: str):
        logging.debug(f"MockSentimentAnalyzer: Analyzing '{text}'")
        if "신나" in text or "기분 좋" in text or "활기찬" in text:
            return {"label": "긍정", "score": 0.9}
        elif "슬프" in text or "우울" in text:
            return {"label": "부정", "score": 0.8}
        elif "공부" in text or "집중" in text or "잔잔" in text:
            return {"label": "긍정", "score": 0.7}
        elif "화가 나" in text or "분노" in text:
            return {"label": "부정", "score": 0.85}
        elif "스트레스" in text or "쉬고 싶" in text:
            return {"label": "부정", "score": 0.75}
        else:
            return {"label": "긍정", "score": 0.5}

class BPMMapper:
    def __init__(self):
        self.emotion_features_map = {
            "긍정": {"bpm": (110, 140), "danceability": (70, 100), "acousticness": (0, 30)},
            "부정": {"bpm": (60, 90), "danceability": (0, 40), "acousticness": (50, 100)},
            "공부": {"bpm": (80, 110), "danceability": (0, 50), "acousticness": (30, 70)},
            "화가 나": {"bpm": (130, 180), "danceability": (50, 90), "acousticness": (0, 20)},
            "스트레스": {"bpm": (70, 100), "danceability": (20, 60), "acousticness": (40, 90)},
            "편안": {"bpm": (70, 100), "danceability": (20, 60), "acousticness": (40, 90)},
            "불안": {"bpm": (95, 125), "danceability": (30, 70), "acousticness": (20, 60)},
            "neutral": {"bpm": (90, 120), "danceability": (40, 80), "acousticness": (20, 70)}
        }
        logging.info("BPM 매퍼 초기화 완료. 기본 오디오 특성 범위: {'bpm': (90, 120), 'danceability': (40, 80), 'acousticness': (20, 70)}")

    def get_audio_feature_ranges(self, emotion_label: str):
        logging.debug(f"BPMMapper: Mapping audio features for '{emotion_label}'")
        return self.emotion_features_map.get(emotion_label, self.emotion_features_map["neutral"])

# --- 실제 SentimentAnalyzer 및 BPMMapper 임포트 시도 ---
try:
    from sentiment_analyzer import SentimentAnalyzer as RealSentimentAnalyzer
    from bpm_mapper import BPMMapper as RealBPMMapper
    
    SentimentAnalyzer = RealSentimentAnalyzer
    BPMMapper = RealBPPMapper
    logging.debug("Actual SentimentAnalyzer and BPMMapper imported successfully.")
except ImportError as e:
    logging.warning(f"Failed to import actual NLP modules: {e}. Continuing with Mock versions.")

# --- 오디오 특성 기반 관련성 점수 계산 함수 (Mock과 실제 모두에서 사용) ---
def _calculate_relevance_score(song_data: dict, target_features: dict) -> float:
    """
    노래의 오디오 특성이 목표 범위에 얼마나 잘 부합하는지 점수를 계산합니다.
    각 특성(BPM, Danceability, Acousticness)은 0~1점 사이의 기여도를 가집니다.
    총 점수는 각 특성 점수의 합계입니다.
    """
    score = 0.0
    weights = {
        "bpm": 1.0,
        "danceability": 1.0,
        "acousticness": 1.0,
    }

    def calculate_feature_score(song_value, min_target, max_target):
        if song_value is None:
            return 0.0
        try:
            # getsong.co API에서 0-100 범위로 반환될 것으로 가정
            song_value = int(song_value) 
            if min_target <= song_value <= max_target:
                center_target = (min_target + max_target) / 2
                range_half = (max_target - min_target) / 2
                if range_half > 0:
                    return (1 - abs(song_value - center_target) / range_half) * 0.5 + 0.5
                else:
                    return 1.0 if song_value == min_target else 0.0
            return 0.0
        except ValueError:
            return 0.0

    score += calculate_feature_score(song_data.get("bpm"), *target_features["bpm"]) * weights["bpm"]
    score += calculate_feature_score(song_data.get("danceability"), *target_features["danceability"]) * weights["danceability"]
    score += calculate_feature_score(song_data.get("acousticness"), *target_features["acousticness"])
    
    return score

class MockMusicRecommender(object):
    """
    API 키가 없거나 개발 시에 사용할 모의(Mock) 추천기입니다.
    실제 API 호출 없이 미리 정의된 데이터를 반환합니다.
    """
    def __init__(self, getsongbpm_api_key: str = None): # getsong_recommendation_api_url 인자 제거
        self.sentiment_analyzer = SentimentAnalyzer()
        self.bpm_mapper = BPMMapper()
        self.getsongbpm_api_key = getsongbpm_api_key # 사용되지 않지만 일관성을 위해 유지
        logging.info("--- Mock Music Recommender initialized (using mock data only) ---")

    def recommend_music(self, user_text: str, limit: int = 3):
        logging.info(f"Mock API call: Simulating recommendation for '{user_text}' with limit {limit}...")
        
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_text)
        user_emotion = sentiment_result["label"]
        target_audio_features = self.bpm_mapper.get_audio_feature_ranges(user_emotion)

        mock_songs_data = [
            {"title": "Dancing Monkey (Mock)", "artist": "Tones And I (Mock)", "bpm": 98, "uri": "#", "genres": ["Pop", "Indie"], "danceability": 80, "acousticness": 10},
            {"title": "Shape of You (Mock)", "artist": "Ed Sheeran (Mock)", "bpm": 96, "uri": "#", "genres": ["Pop", "R&B"], "danceability": 85, "acousticness": 5},
            {"title": "Blinding Lights (Mock)", "artist": "The Weeknd (Mock)", "bpm": 171, "uri": "#", "genres": ["Pop", "Synth-pop"], "danceability": 75, "acousticness": 2},
            {"title": "Dynamite (Mock)", "artist": "BTS (Mock)", "bpm": 114, "uri": "#", "genres": ["K-Pop", "Disco-Pop"], "danceability": 90, "acousticness": 1},
            {"title": "Bad Guy (Mock)", "artist": "Billie Eilish (Mock)", "bpm": 135, "uri": "#", "genres": ["Pop", "Electropop"], "danceability": 70, "acousticness": 15},
            {"title": "Old Town Road (Mock)", "artist": "Lil Nas X (Mock)", "bpm": 136, "uri": "#", "genres": ["Country Rap"], "danceability": 65, "acousticness": 20},
            {"title": "Someone You Loved (Mock)", "artist": "Lewis Capaldi (Mock)", "bpm": 109, "uri": "#", "genres": ["Pop", "Ballad"], "danceability": 30, "acousticness": 80},
            {"title": "Happy (Mock)", "artist": "Pharrell Williams (Mock)", "bpm": 160, "uri": "#", "genres": ["Pop", "Soul"], "danceability": 95, "acousticness": 5},
            {"title": "Uptown Funk (Mock)", "artist": "Mark Ronson (Mock)", "bpm": 115, "uri": "#", "genres": ["Funk", "Pop"], "danceability": 88, "acousticness": 8},
            {"title": "Bohemian Rhapsody (Mock)", "artist": "Queen (Mock)", "bpm": 144, "uri": "#", "genres": ["Rock", "Classic Rock"], "danceability": 40, "acousticness": 30},
            {"title": "Lose Yourself (Mock)", "artist": "Eminem (Mock)", "bpm": 171, "uri": "#", "genres": ["Hip Hop", "Rap"], "danceability": 60, "acousticness": 10},
            {"title": "Imagine (Mock)", "artist": "John Lennon (Mock)", "bpm": 75, "uri": "#", "genres": ["Pop", "Soft Rock"], "danceability": 20, "acousticness": 90},
            {"title": "What a Wonderful World (Mock)", "artist": "Louis Armstrong (Mock)", "bpm": 82, "uri": "#", "genres": ["Jazz", "Vocal"], "danceability": 35, "acousticness": 70},
            {"title": "Stairway to Heaven (Mock)", "artist": "Led Zeppelin (Mock)", "bpm": 147, "uri": "#", "genres": ["Rock", "Hard Rock"], "danceability": 25, "acousticness": 60},
            {"title": "Hotel California (Mock)", "artist": "Eagles (Mock)", "bpm": 147, "uri": "#", "genres": ["Rock", "Classic Rock"], "danceability": 50, "acousticness": 40},
            {"title": "Yesterday (Mock)", "artist": "The Beatles (Mock)", "bpm": 94, "uri": "#", "genres": ["Pop", "Rock"], "danceability": 45, "acousticness": 50},
            {"title": "Smells Like Teen Spirit (Mock)", "artist": "Nirvana (Mock)", "bpm": 117, "uri": "#", "genres": ["Grunge", "Rock"], "danceability": 60, "acousticness": 10},
            {"title": "Billie Jean (Mock)", "artist": "Michael Jackson (Mock)", "bpm": 117, "uri": "#", "genres": ["Pop", "Funk"], "danceability": 90, "acousticness": 5},
            {"title": "Like a Rolling Stone (Mock)", "artist": "Bob Dylan (Mock)", "bpm": 95, "uri": "#", "genres": ["Folk Rock"], "danceability": 40, "acousticness": 70},
            {"title": "One (Mock)", "artist": "U2 (Mock)", "bpm": 91, "uri": "#", "genres": ["Rock"], "danceability": 35, "acousticness": 60},
        ]
            
        unique_mock_songs = []
        mock_seen_titles = set()
        
        common_titles_to_filter = [
            "tune", "mix", "track", "edit", "version", "remix", "instrumental",
            "intro", "outro", "interlude", "skit", "freestyle", "demo",
            "live", "acoustic", "radio", "original", "album", "single",
            "theme", "song", "beat", "rhythm", "sound", "music", "vocal",
            "dance", "pop", "rock", "jazz", "hip hop", "electronic", "ballad", "soul", "funk", "classical",
            "upbeat", "energetic", "optimistic", "happy", "sad", "melancholy", "calm", "chill", "relaxed",
            "part", "chapter", "episode", "vol", "volume", "session", "loop", "medley"
        ]
        common_titles_to_filter_lower = [word.lower() for word in common_titles_to_filter]

        for song in mock_songs_data:
            is_mock_title_a_genre = False
            if song.get("genres") and song["title"].lower() in [g.lower() for g in song["genres"]]:
                is_mock_title_a_genre = True

            is_mock_title_common_word = False
            if song["title"].lower() in common_titles_to_filter_lower:
                is_mock_title_common_word = True
            elif any(song["title"].lower().startswith(word) and song["title"][len(word):].strip().isdigit() for word in common_titles_to_filter_lower):
                is_mock_title_common_word = True

            if song["title"] not in mock_seen_titles and not is_mock_title_a_genre and not is_mock_title_common_word:
                relevance_score = _calculate_relevance_score( # 전역 함수 호출
                    {"bpm": song["bpm"], "danceability": song["danceability"], "acousticness": song["acousticness"]},
                    target_audio_features
                )
                song["relevance_score"] = relevance_score
                unique_mock_songs.append(song)
                mock_seen_titles.add(song["title"])
            
        unique_mock_songs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        recommended_songs = unique_mock_songs[:limit]

        return {
            "user_emotion": user_emotion,
            "target_audio_features": target_audio_features,
            "recommendations": recommended_songs
        }


class MusicRecommender:
    """
    사용자의 감정을 분석하여 음악 특성 기반으로 음악을 추천하는 클래스입니다.
    """
    def __init__(self, getsongbpm_api_key: str): # getsong_recommendation_api_url 인자 제거
        """
        MusicRecommender를 초기화합니다.

        Args:
            getsongbpm_api_key (str): getsong.co API 키.
        """
        self.sentiment_analyzer = SentimentAnalyzer()
        self.bpm_mapper = BPMMapper()
        self.getsongbpm_api_key = getsongbpm_api_key
        # Getsong API의 기본 URL과 텍스트 추천 엔드포인트를 내부에 정의
        self.getsongbpm_base_url = "https://api.getsong.co/" 
        # TODO: Getsong API 문서에서 텍스트 기반 추천의 정확한 엔드포인트를 확인하여 교체하세요.
        self.text_recommendation_endpoint = "v1/text-recommendation" # 예시 URL, 실제 URL로 교체 필요
        self.getsong_recommendation_api_url = f"{self.getsongbpm_base_url}{self.text_recommendation_endpoint}"

        logging.info(f"Music recommendation system initialized with API URL: {self.getsong_recommendation_api_url}")

    def recommend_music(self, user_text: str, limit: int = 3):
        """
        사용자 텍스트를 기반으로 음악을 추천하는 메인 메서드입니다.
        감정 분석, 오디오 특성 매핑, Getsong API 호출을 포함합니다.
        """
        logging.info(f"Starting recommendation process for user text: '{user_text}'")
        
        # 1. 감정 분석
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_text)
        user_emotion = sentiment_result["label"]
        logging.debug(f"User emotion analyzed as: {user_emotion}")
        
        # 2. 오디오 특성 범위 매핑 (API에 전달할 필요는 없을 수 있지만, 설명용으로 유지)
        target_audio_features = self.bpm_mapper.get_audio_feature_ranges(user_emotion)
        logging.debug(f"Target audio features for '{user_emotion}': {target_audio_features}")
        
        # 3. Getsong API 호출
        headers = {
            "Content-Type": "application/json"
        }
        # Getsong API의 요청 바디 형식에 맞게 조정해야 합니다.
        # 일반적으로 텍스트 기반 추천 API는 텍스트 쿼리, 감정, 원하는 추천 개수 등을 받습니다.
        payload = {
            "query": user_text,
            "emotion": user_emotion, # 분석된 감정도 함께 전달 (API가 지원한다면)
            "limit": limit,
            "api_key": self.getsongbpm_api_key # API 키를 바디에 포함하는 경우 (getsong.co 문서 확인)
        }
        # 또는 API 키를 URL 파라미터로 전달하는 경우 (getsong.co 문서 확인)
        # params = {"api_key": self.getsongbpm_api_key, "query": user_text, "emotion": user_emotion, "limit": limit}

        try:
            # Getsong API 문서에 따라 GET 또는 POST 요청 사용
            # 텍스트 기반 추천은 보통 POST 요청입니다.
            response = requests.post(self.getsong_recommendation_api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status() # HTTP 오류가 발생하면 예외 발생

            api_data = response.json()
            logging.info(f"Getsong API 응답 수신: {json.dumps(api_data, indent=2, ensure_ascii=False)}")

            # Getsong API의 실제 응답 구조에 따라 데이터를 파싱하고 변환해야 합니다.
            # 예시: API 응답에 'recommended_tracks' 또는 'results'와 같은 키가 있다고 가정
            raw_recommendations = api_data.get("recommended_tracks", api_data.get("results", []))

            recommendations = []
            for track in raw_recommendations:
                # API 응답 필드명을 HTML에서 사용하는 필드명으로 매핑
                # 실제 Getsong API 응답 필드명에 맞춰 수정 필요
                recommendations.append({
                    "title": track.get("title", "알 수 없는 제목"),
                    "artist": track.get("artist_name", "알 수 없는 아티스트"), # 예시: artist_name
                    "uri": track.get("youtube_url", "#"), # Getsong API가 Youtube URL을 직접 제공하는 경우
                    "genres": track.get("genres", []),
                    "bpm": track.get("tempo", 0),
                    "danceability": track.get("danceability", 0) * 100, # 0-1 값을 0-100으로 변환
                    "acousticness": track.get("acousticness", 0) * 100, # 0-1 값을 0-100으로 변환
                    "relevance_score": track.get("relevance_score", 0) # API에서 제공하는 적합성 점수
                })
            
            # API에서 점수 정렬이 안 되어 있다면 여기서 다시 정렬
            recommendations.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            return {
                "user_emotion": user_emotion, # 분석된 감정
                "target_audio_features": target_audio_features, # 매핑된 특성 범위
                "recommendations": recommendations # 최종 추천곡 목록
            }

        except requests.exceptions.Timeout:
            logging.error("Getsong API 요청 시간 초과.")
            raise Exception("Getsong API 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Getsong API 요청 중 네트워크 또는 HTTP 오류 발생: {e}")
            if "404 Client Error" in str(e):
                raise Exception(f"Getsong API 엔드포인트 URL이 잘못되었거나 존재하지 않습니다: {self.getsong_recommendation_api_url}")
            else:
                raise Exception(f"Getsong API 통신 중 오류가 발생했습니다: {str(e)}")
        except json.JSONDecodeError:
            logging.error(f"Getsong API 응답 JSON 파싱 실패: {response.text}")
            raise Exception("Getsong API 응답 형식이 올바르지 않습니다.")
        except Exception as e:
            logging.error(f"Getsong API 응답 처리 중 알 수 없는 오류 발생: {e}")
            raise Exception(f"음악 추천 데이터를 처리하는 중 오류가 발생했습니다: {str(e)}")


# 이 파일이 직접 실행될 때만 실행되는 테스트 코드
if __name__ == "__main__":
    # 환경 변수에서 API 키를 가져옵니다.
    getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY")

    if getsongbpm_api_key:
        logging.info("GETSONGBPM_API_KEY가 설정되어 실제 API를 사용합니다.")
        recommender = MusicRecommender(getsongbpm_api_key)
    else:
        logging.warning("GETSONGBPM_API_KEY 환경 변수가 설정되지 않았습니다. Mock 추천기를 사용합니다.")
        recommender = MockMusicRecommender()

    logging.info("\n==== Music Recommendation System Demo Start ====")

    test_inputs = [
        "오늘은 정말 기분이 좋고 신나!",
        "요즘 너무 우울해서 슬픈 노래가 듣고 싶어.",
        "공부해야 하는데 집중이 안 돼. 잔잔한 음악 틀어줘.",
        "정말 화가 나서 아무것도 못 하겠어.",
        "하루 종일 스트레스 받아서 쉬고 싶어.",
        "너무 편안하고 기분이 좋아.",
        "조금 불안하고 걱정이 되네."
    ]

    for user_text in test_inputs:
        try:
            result = recommender.recommend_music(user_text)
            logging.info(f"\nUser Emotion: {result['user_emotion']}")
            logging.info(f"Target Audio Features: {result['target_audio_features']}")
            logging.info("Recommended Songs:")
            for song in result['recommendations']:
                logging.info(f"  - {song['title']} by {song['artist']} (Score: {song.get('relevance_score', 'N/A'):.2f})")
            logging.info("\n" + "="*70 + "\n")
        except Exception as e:
            logging.error(f"테스트 중 오류 발생: {e}")
            logging.info("\n" + "="*70 + "\n")
