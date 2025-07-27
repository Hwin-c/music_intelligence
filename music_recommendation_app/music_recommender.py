import requests
import random
import os
import sys
import logging
import time
import traceback
import json

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# natural-language-processing 디렉토리를 Python 경로에 추가하여 모듈을 임포트할 수 있도록 합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
nlp_dir = os.path.join(current_dir, 'natural-language-processing')

if nlp_dir not in sys.path:
    sys.path.insert(0, nlp_dir)
    logging.debug(f"Added {nlp_dir} to sys.path for NLP modules.")

# --- Mock SentimentAnalyzer 및 BPMMapper 클래스들을 먼저 정의합니다. ---
# 이 클래스들은 실제 모듈 임포트가 실패할 경우 사용될 fallback 입니다.
class SentimentAnalyzer: # 기본적으로 Mock 버전으로 시작
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

class BPMMapper: # 기본적으로 Mock 버전으로 시작
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
        logging.info("MockBPMMapper initialized.")

    def get_audio_feature_ranges(self, emotion_label: str):
        logging.debug(f"MockBPMMapper: Mapping audio features for '{emotion_label}'")
        return self.emotion_features_map.get(emotion_label, self.emotion_features_map["neutral"])

# --- 실제 SentimentAnalyzer 및 BPMMapper 임포트 시도 ---
# 성공하면 위에 정의된 Mock 클래스들을 덮어씁니다.
try:
    from sentiment_analyzer import SentimentAnalyzer as RealSentimentAnalyzer
    from bpm_mapper import BPMMapper as RealBPMMapper
    
    # 실제 클래스로 전역 변수를 덮어씁니다.
    SentimentAnalyzer = RealSentimentAnalyzer
    BPMMapper = RealBPMMapper
    logging.debug("Actual SentimentAnalyzer and BPMMapper imported successfully.")
except ImportError as e:
    logging.warning(f"Failed to import actual NLP modules: {e}. Continuing with Mock versions.")
    # 이 경우 SentimentAnalyzer와 BPMMapper는 이미 위에 정의된 Mock 버전입니다.


class MusicRecommender:
    """
    사용자의 감정을 분석하여 음악 특성 기반으로 음악을 추천하는 클래스입니다.
    """
    def __init__(self, getsongbpm_api_key: str):
        """
        MusicRecommender를 초기화합니다.

        Args:
            getsongbpm_api_key (str): getsong.co API 키.
        """
        self.sentiment_analyzer = SentimentAnalyzer() # 이제 SentimentAnalyzer는 실제 또는 Mock 중 하나
        self.bpm_mapper = BPMMapper() # 이제 BPMMapper는 실제 또는 Mock 중 하나
        self.getsongbpm_api_key = getsongbpm_api_key
        self.getsongbpm_base_url = "https://api.getsong.co/" 
        logging.info(f"Music recommendation system initialized with API Base URL: {self.getsongbpm_base_url}")

    def _call_getsongbpm_api(self, endpoint: str, params: dict = None, delay_seconds: float = 1.5):
        """
        getsong.co API를 호출하는 내부 도우미 메서드입니다.
        API Key는 URL 파라미터로 전송합니다.
        Rate Limit를 피하기 위해 호출 사이에 지연을 추가합니다.
        """
        if params is None:
            params = {}
        params["api_key"] = self.getsongbpm_api_key 
        
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]
        
        url = f"{self.getsongbpm_base_url}{endpoint}"
        
        try:
            logging.debug(f"Calling getsong.co API: {url} with params {params}")
            response = requests.get(url, params=params)
            response.raise_for_status() 
            
            response_json = response.json()
            logging.debug(f"getsong.co API response JSON: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
            
            return response_json
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err} - Response Status: {response.status_code if response else 'N/A'} - Response Text: {response.text if response else 'N/A'}")
        except requests.exceptions.ConnectionError as conn_err:
            logging.error(f"Network connection error: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logging.error(f"Request timeout error: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logging.error(f"Request error: {req_err}")
        finally:
            time.sleep(delay_seconds) # API 호출 후 지연 추가
        return None

    def _calculate_relevance_score(self, song_data: dict, target_features: dict) -> float:
        """
        노래의 오디오 특성이 목표 범위에 얼마나 잘 부합하는지 점수를 계산합니다.
        각 특성(BPM, Danceability, Acousticness)은 0~1점 사이의 기여도를 가집니다.
        총 점수는 각 특성 점수의 합계입니다.

        점수 계산 로직:
        1. 특성 값이 목표 범위 내에 있으면 기본 점수 (0.5점) 부여.
        2. 특성 값이 목표 범위의 중앙값에 가까울수록 추가 점수 (최대 0.5점) 부여.
           - 범위 중앙과의 거리가 멀수록 점수 감소.
           - 범위가 0인 경우 (min=max)는 값이 일치하면 1.0점, 아니면 0점.
        3. 각 특성 점수의 합이 최종 점수.
        """
        score = 0.0

        # 특성별 가중치 (필요시 조정)
        weights = {
            "bpm": 1.0,
            "danceability": 1.0,
            "acousticness": 1.0
        }

        # BPM 점수 계산
        song_bpm = song_data.get("tempo")
        min_bpm, max_bpm = target_features["bpm"]
        if song_bpm is not None:
            try:
                song_bpm = int(song_bpm)
                if min_bpm <= song_bpm <= max_bpm:
                    center_bpm = (min_bpm + max_bpm) / 2
                    range_half = (max_bpm - min_bpm) / 2
                    if range_half > 0:
                        score_component = (1 - abs(song_bpm - center_bpm) / range_half) * 0.5 + 0.5
                    else:
                        score_component = 1.0 if song_bpm == min_bpm else 0.0
                    score += score_component * weights["bpm"]
            except ValueError:
                pass

        # Danceability 점수 계산
        song_danceability = song_data.get("danceability")
        min_dance, max_dance = target_features["danceability"]
        if song_danceability is not None:
            try:
                song_danceability = int(song_danceability)
                if min_dance <= song_danceability <= max_dance:
                    center_dance = (min_dance + max_dance) / 2
                    range_half = (max_dance - min_dance) / 2
                    if range_half > 0:
                        score_component = (1 - abs(song_danceability - center_dance) / range_half) * 0.5 + 0.5
                    else:
                        score_component = 1.0 if song_danceability == min_dance else 0.0
                    score += score_component * weights["danceability"]
            except ValueError:
                pass

        # Acousticness 점수 계산
        song_acousticness = song_data.get("acousticness")
        min_acoustic, max_acoustic = target_features["acousticness"]
        if song_acousticness is not None:
            try:
                song_acousticness = int(song_acousticness)
                if min_acoustic <= song_acousticness <= max_acoustic:
                    center_acoustic = (min_acoustic + max_acoustic) / 2
                    range_half = (max_acoustic - min_acoustic) / 2
                    if range_half > 0:
                        score_component = (1 - abs(song_acousticness - center_acoustic) / range_half) * 0.5 + 0.5
                    else:
                        score_component = 1.0 if song_acousticness == min_acoustic else 0.0
                    score += score_component * weights["acousticness"]
            except ValueError:
                pass
        
        return score

    def get_ranked_songs_by_audio_features(self, emotion_label: str, limit: int = 3):
        """
        사용자의 감정에 따라 오디오 특성 기반으로 노래를 검색하고 랭킹을 매겨 반환합니다.
        '/search' 엔드포인트를 활용하여 더 다양한 제목을 얻도록 시도합니다.
        """
        logging.info(f"Attempting to search and rank songs based on audio features for '{emotion_label}' emotion via getsong.co API...")
        
        target_features = self.bpm_mapper.get_audio_feature_ranges(emotion_label)
        
        candidate_songs = []
        seen_titles = set() # 중복 제목을 추적하기 위한 셋

        # 제목에서 필터링할 일반적인 단어 리스트 추가 및 소문자 변환
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

        # 감정 기반 검색 키워드 매핑 (더 다양하고 구체적인 키워드 추가)
        emotion_keywords_for_search = {
            "긍정": ["happy", "upbeat", "energetic", "joyful", "party", "celebration", "optimistic", "bright", "good vibes", "dancing", "fun"],
            "부정": ["sad", "melancholy", "downbeat", "gloomy", "depressed", "lonely", "heartbreak", "somber", "blue"],
            "공부": ["focus", "study", "concentration", "ambient", "instrumental", "classical", "lo-fi", "calm", "relaxing"],
            "화가 나": ["angry", "rage", "intense", "aggressive", "metal", "punk", "hard rock", "rebellion"],
            "스트레스": ["relax", "calm", "chill", "soothing", "meditation", "peaceful", "unwind"],
            "편안": ["relax", "calm", "chill", "peaceful", "smooth", "mellow", "serene"],
            "불안": ["soothing", "calm", "meditation", "peaceful", "gentle", "comforting"],
            "neutral": ["easy listening", "background music", "chill out", "acoustic", "mellow"]
        }
        
        # 일반적인 장르 키워드 (fallback 용도 또는 추가 다양성)
        genre_keywords = ["pop", "dance", "rock", "electronic", "jazz", "hip hop", "ballad", "r&b", "k-pop", "indie", "soul", "funk", "classical"]

        # 감정 기반 키워드를 우선 사용하고, 장르 키워드를 추가합니다.
        combined_queries = set(emotion_keywords_for_search.get(emotion_label, emotion_keywords_for_search["neutral"]))
        combined_queries.update(genre_keywords)
        
        search_queries_list = list(combined_queries)
        random.shuffle(search_queries_list)

        max_api_calls = 5 # 최대 API 호출 횟수 제한 (rate limit 고려)
        calls_made = 0

        for query in search_queries_list:
            if len(candidate_songs) >= limit * 5 and calls_made >= max_api_calls:
                break

            params = {"type": "song", "lookup": query, "limit": 50}
            api_response = self._call_getsongbpm_api("search/", params, delay_seconds=1.5)
            calls_made += 1

            if api_response and api_response.get("search"): 
                for song_data in api_response["search"]: 
                    song_title = song_data.get("title", "Unknown Title")
                    song_uri = song_data.get("uri", "#") 
                    
                    artist_name = "Unknown Artist"
                    artist_info = song_data.get("artist") 
                    if isinstance(artist_info, dict): 
                        artist_name = artist_info.get("name", "Unknown Artist")
                    
                    genres = []
                    if isinstance(artist_info, dict):
                        artist_genres = artist_info.get("genres")
                        if isinstance(artist_genres, list):
                            genres = artist_genres
                        else:
                            genres = []
                    else:
                        genres = []

                    # 1. 이미 추가된 제목인지 확인하여 중복 방지
                    if song_title in seen_titles:
                        continue 

                    # 2. 노래 제목이 장르 이름과 동일한지 확인하여 필터링
                    is_title_a_genre = False
                    for g in genres:
                        if song_title.lower() == g.lower():
                            is_title_a_genre = True
                            break
                    if is_title_a_genre:
                        continue

                    # 3. 노래 제목이 일반적인 단어인지 확인하여 필터링
                    is_title_common_word = False
                    # 제목 전체가 일반적인 단어인 경우
                    if song_title.lower() in common_titles_to_filter_lower:
                        is_title_common_word = True
                    # 제목이 "Tune 2"와 같이 일반적인 단어로 시작하고 뒤에 숫자가 붙는 경우
                    elif any(song_title.lower().startswith(word) and song_title[len(word):].strip().isdigit() for word in common_titles_to_filter_lower):
                        is_title_common_word = True
                    
                    if is_title_common_word:
                        logging.debug(f"Filtering out song with common title: {song_title}")
                        continue

                    song_bpm = song_data.get("tempo") 
                    song_danceability = song_data.get("danceability")
                    song_acousticness = song_data.get("acousticness")

                    if song_bpm is not None and song_danceability is not None and song_acousticness is not None:
                        try:
                            song_bpm = int(song_bpm)
                            song_danceability = int(song_danceability)
                            song_acousticness = int(song_acousticness)

                            relevance_score = self._calculate_relevance_score(
                                {"tempo": song_bpm, "danceability": song_danceability, "acousticness": song_acousticness},
                                target_features
                            )

                            if relevance_score > 0:
                                candidate_songs.append({
                                    "title": song_title,
                                    "artist": artist_name,
                                    "bpm": song_bpm,
                                    "uri": song_uri, 
                                    "genres": genres,
                                    "danceability": song_danceability,
                                    "acousticness": song_acousticness,
                                    "relevance_score": relevance_score
                                })
                                seen_titles.add(song_title)
                        except ValueError:
                            logging.warning(f"Invalid numeric value for audio feature in song {song_title}. Skipping.")
                    else:
                        logging.debug(f"Missing or invalid audio features for song {song_title}. Skipping for ranking.")
            
            if len(candidate_songs) >= limit * 5:
                break

        candidate_songs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        recommended_songs = candidate_songs[:limit]

        if not recommended_songs:
            logging.warning(f"No relevant songs found from getsong.co API after filtering and ranking. Falling back to mock data.")
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
                {"title": "Stairway to Heaven (Mock)", "artist": "Led Zeppelin (Mock)", "bpm": 147, "uri": "#", "genres": ["Rock", "Hard Rock"], "danceability": 25, "acousticness": 60}, # BPM 82 -> 147로 변경
                {"title": "Hotel California (Mock)", "artist": "Eagles (Mock)", "bpm": 147, "uri": "#", "genres": ["Rock", "Classic Rock"], "danceability": 50, "acousticness": 40},
                {"title": "Yesterday (Mock)", "artist": "The Beatles (Mock)", "bpm": 94, "uri": "#", "genres": ["Pop", "Rock"], "danceability": 45, "acousticness": 50},
                {"title": "Smells Like Teen Spirit (Mock)", "artist": "Nirvana (Mock)", "bpm": 117, "uri": "#", "genres": ["Grunge", "Rock"], "danceability": 60, "acousticness": 10},
                {"title": "Billie Jean (Mock)", "artist": "Michael Jackson (Mock)", "bpm": 117, "uri": "#", "genres": ["Pop", "Funk"], "danceability": 90, "acousticness": 5},
                {"title": "Like a Rolling Stone (Mock)", "artist": "Bob Dylan (Mock)", "bpm": 95, "uri": "#", "genres": ["Folk Rock"], "danceability": 40, "acousticness": 70},
                {"title": "One (Mock)", "artist": "U2 (Mock)", "bpm": 91, "uri": "#", "genres": ["Rock"], "danceability": 35, "acousticness": 60},
            ]
            
            unique_mock_songs = []
            mock_seen_titles = set()
            
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
                    relevance_score = self._calculate_relevance_score(
                        {"tempo": song["bpm"], "danceability": song["danceability"], "acousticness": song["acousticness"]},
                        target_features
                    )
                    song["relevance_score"] = relevance_score
                    unique_mock_songs.append(song)
                    mock_seen_titles.add(song["title"])
            
            unique_mock_songs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return unique_mock_songs[:limit]


# 이 파일이 직접 실행될 때만 실행되는 테스트 코드
if __name__ == "__main__":
    getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY", "YOUR_GETSONGBPM_API_KEY_HERE")

    if getsongbpm_api_key == "YOUR_GETSONGBPM_API_KEY_HERE":
        logging.warning("\n[WARNING]: getsongbpm API key is not set. Proceeding with mock data for testing.")
        recommender = MockMusicRecommender(getsongbpm_api_key)
    else:
        logging.info("GETSONGBPM_API_KEY is set. Proceeding with actual API calls.")
        recommender = MusicRecommender(getsongbpm_api_key)

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
        result = recommender.recommend_music(user_text)
        logging.info(f"\nUser Emotion: {result['user_emotion']}")
        logging.info(f"Target Audio Features: {result['target_audio_features']}")
        logging.info("\n" + "="*70 + "\n")