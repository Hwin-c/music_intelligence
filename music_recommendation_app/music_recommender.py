import requests
import random
import os
import sys
import logging
import time

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock SentimentAnalyzer 및 BPMMapper 클래스 (실제 모듈이 없을 경우를 대비)
class MockSentimentAnalyzer:
    def analyze_sentiment(self, text: str):
        logging.debug(f"MockSentimentAnalyzer: Analyzing '{text}'")
        if "신나" in text or "기분 좋" in text or "활기찬" in text:
            return {"label": "긍정", "score": 0.9} # Mock에서도 '긍정'/'부정' 반환
        elif "슬프" in text or "우울" in text:
            return {"label": "부정", "score": 0.8} # Mock에서도 '긍정'/'부정' 반환
        elif "공부" in text or "집중" in text or "잔잔" in text:
            return {"label": "긍정", "score": 0.7} # 집중도 긍정적인 상태로 간주
        elif "화가 나" in text or "분노" in text:
            return {"label": "부정", "score": 0.85}
        elif "스트레스" in text or "쉬고 싶" in text:
            return {"label": "부정", "score": 0.75}
        else:
            return {"label": "긍정", "score": 0.5} # 기본값도 '긍정'으로 설정하여 BPM 매핑되도록

class MockBPMMapper:
    def __init__(self):
        self.bpm_map = {
            "긍정": (110, 140), # SentimentAnalyzer의 '긍정'에 매핑
            "부정": (60, 90),  # SentimentAnalyzer의 '부정'에 매핑
            "happy": (120, 160), # 기존 매핑 유지 (혹시 모를 경우 대비)
            "sad": (60, 90),
            "calm": (80, 110),
            "angry": (130, 180),
            "relaxed": (70, 100),
            "anxious": (95, 125),
            "neutral": (90, 120)
        }
        logging.info("MockBPMMapper initialized.")

    def get_bpm_range(self, emotion_label: str):
        logging.debug(f"MockBPMMapper: Mapping BPM for '{emotion_label}'")
        return self.bpm_map.get(emotion_label, self.bpm_map["neutral"])


# 부모 디렉토리를 Python 경로에 추가하여 natural_language_processing 모듈을 임포트할 수 있도록 합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
nlp_dir = os.path.join(current_dir, 'natural-language-processing')

if nlp_dir not in sys.path:
    sys.path.append(nlp_dir)
    logging.debug(f"Added {nlp_dir} to sys.path")

try:
    # 실제 SentimentAnalyzer 및 BPMMapper 임포트 시도
    from sentiment_analyzer import SentimentAnalyzer
    from bpm_mapper import BPMMapper
    logging.debug("SentimentAnalyzer and BPMMapper imported successfully.")
except ImportError as e:
    logging.warning(f"Failed to import actual NLP modules: {e}. Using Mock versions.")
    # 실제 모듈 임포트 실패 시 Mock 버전 사용
    SentimentAnalyzer = MockSentimentAnalyzer
    BPMMapper = MockBPMMapper


class MusicRecommender:
    """
    사용자의 감정을 분석하여 BPM 기반으로 음악을 추천하는 클래스입니다.
    """
    def __init__(self, getsongbpm_api_key: str):
        """
        MusicRecommender를 초기화합니다.

        Args:
            getsongbpm_api_key (str): getsongbpm.com API 키.
        """
        self.sentiment_analyzer = SentimentAnalyzer()
        self.bpm_mapper = BPMMapper()
        self.getsongbpm_api_key = getsongbpm_api_key
        self.getsongbpm_base_url = "https://api.getsongbpm.com"
        logging.info("Music recommendation system initialized.")

    def _call_getsongbpm_api(self, endpoint: str, params: dict = None):
        """
        getsongbpm API를 호출하는 내부 도우미 메서드입니다.
        """
        headers = {
            "x-api-key": self.getsongbpm_api_key,
            "Content-Type": "application/json"
        }
        
        url = f"{self.getsongbpm_base_url}{endpoint}"
        try:
            logging.debug(f"Calling getsongbpm API: {url} with params {params}")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err} - Response Text: {response.text if response else 'N/A'}")
        except requests.exceptions.ConnectionError as conn_err:
            logging.error(f"Network connection error: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logging.error(f"Request timeout error: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logging.error(f"Request error: {req_err}")
        return None

    def get_songs_by_bpm_range(self, min_bpm: int, max_bpm: int, limit: int = 5):
        """
        getsongbpm API를 사용하여 특정 BPM 범위 내의 노래를 검색합니다.
        실제 API 호출을 시도하고, 실패 시 Mock 데이터로 대체합니다.
        """
        logging.info(f"Attempting to search for songs in BPM range {min_bpm}~{max_bpm} using getsongbpm API...")
        
        found_songs = []
        try:
            # getsongbpm API의 /search/tracks 엔드포인트를 사용합니다.
            # 직접적인 BPM 범위 검색은 지원하지 않으므로, 일반적인 검색어(예: "pop", "dance")로 검색 후 클라이언트 측에서 필터링합니다.
            # 더 많은 결과를 얻기 위해 여러 일반적인 장르/키워드를 시도할 수 있습니다.
            search_queries = ["pop", "dance", "rock", "electronic", "jazz", "hip hop"]
            
            for query in search_queries:
                params = {"q": query, "per_page": 20} # 페이지당 더 많은 결과 요청
                api_response = self._call_getsongbpm_api("/search/tracks", params)

                if api_response and api_response.get("tracks"):
                    for track in api_response["tracks"]:
                        track_bpm = track.get("bpm")
                        if track_bpm is not None:
                            try:
                                track_bpm = int(track_bpm) # BPM이 문자열일 수 있으므로 int로 변환
                                if min_bpm <= track_bpm <= max_bpm:
                                    found_songs.append({
                                        "title": track.get("title", "Unknown Title"),
                                        "artist": track.get("artist", {}).get("name", "Unknown Artist"),
                                        "bpm": track_bpm,
                                        "album_cover_url": track.get("album", {}).get("image", "https://placehold.co/140x140/cccccc/000000?text=No+Cover") # 기본 이미지 제공
                                    })
                                    if len(found_songs) >= limit: # 필요한 개수만큼 찾으면 중단
                                        break
                            except ValueError:
                                logging.warning(f"Invalid BPM value received for track {track.get('title')}: {track_bpm}")
                if len(found_songs) >= limit:
                    break

        except Exception as e:
            logging.error(f"Error during getsongbpm API search: {e}")
            logging.error(traceback.format_exc())
            # API 호출 중 오류가 발생하면 Mock 데이터로 대체
            pass # 아래 로직에서 Mock 데이터로 대체될 것임

        if not found_songs:
            logging.warning(f"No relevant songs found from getsongbpm API for BPM range {min_bpm}~{max_bpm}. Falling back to mock data.")
            # API에서 결과를 찾지 못했을 때만 Mock 데이터로 대체
            mock_songs_data = [
                {"title": "기분 좋은 아침 (Mock)", "artist": "김미소 (Mock)", "bpm": 130, "album_cover_url": "https://placehold.co/140x140/FFD700/000000?text=Happy"},
                {"title": "고요한 숲길 (Mock)", "artist": "이평화 (Mock)", "bpm": 75, "album_cover_url": "https://placehold.co/140x140/ADD8E6/000000?text=Calm"},
                {"title": "집중의 순간 (Mock)", "artist": "박몰입 (Mock)", "bpm": 100, "album_cover_url": "https://placehold.co/140x140/90EE90/000000?text=Focus"},
                {"title": "파워 업! (Mock)", "artist": "최에너지 (Mock)", "bpm": 155, "album_cover_url": "https://placehold.co/140x140/FF4500/FFFFFF?text=Power"},
                {"title": "차분한 저녁 (Mock)", "artist": "정고요 (Mock)", "bpm": 60, "album_cover_url": "https://placehold.co/140x140/8A2BE2/FFFFFF?text=Evening"},
                {"title": "활기찬 하루 (Mock)", "artist": "강다이나믹 (Mock)", "bpm": 120, "album_cover_url": "https://placehold.co/140x140/00CED1/FFFFFF?text=Dynamic"},
                {"title": "생각의 흐름 (Mock)", "artist": "윤명상 (Mock)", "bpm": 90, "album_cover_url": "https://placehold.co/140x140/DDA0DD/000000?text=Thought"},
                {"title": "분노의 질주 (Mock)", "artist": "서파워 (Mock)", "bpm": 140, "album_cover_url": "https://placehold.co/140x140/B22222/FFFFFF?text=Rage"},
                {"title": "슬픈 빗소리 (Mock)", "artist": "오감성 (Mock)", "bpm": 70, "album_cover_url": "https://placehold.co/140x140/6A5ACD/FFFFFF?text=Sad"},
                {"title": "새로운 도전 (Mock)", "artist": "조열정 (Mock)", "bpm": 115, "album_cover_url": "https://placehold.co/140x140/FF8C00/000000?text=Challenge"},
                {"title": "행복한 발걸음 (Mock)", "artist": "김행복 (Mock)", "bpm": 125, "album_cover_url": "https://placehold.co/140x140/FFC0CB/000000?text=Walk"},
                {"title": "꿈꾸는 밤 (Mock)", "artist": "이몽환 (Mock)", "bpm": 85, "album_cover_url": "https://placehold.co/140x140/4682B4/FFFFFF?text=Dream"},
                {"title": "에너지 폭발 (Mock)", "artist": "박다이나믹 (Mock)", "bpm": 160, "album_cover_url": "https://placehold.co/140x140/FF6347/FFFFFF?text=Blast"},
            ]
            return random.sample(mock_songs_data, min(limit, len(mock_songs_data)))
        
        # API에서 찾은 곡이 있다면 그 중에서 랜덤으로 limit 개수만큼 선택
        return random.sample(found_songs, min(limit, len(found_songs)))


    def recommend_music(self, user_text: str):
        """
        사용자 텍스트를 기반으로 음악을 추천합니다.

        Args:
            user_text (str): 사용자의 감정 표현 텍스트.

        Returns:
            list: 추천된 음악 리스트 (제목, 아티스트, BPM, 앨범 커버 URL 포함).
        """
        logging.info(f"\n--- Starting music recommendation for '{user_text}' ---")
        
        # 1. 감정 분석
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_text)
        emotion_label = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]
        
        logging.info(f"Sentiment analysis result: '{emotion_label}' (score: {sentiment_score:.4f})")

        # 2. 감정에 따른 BPM 범위 매핑
        min_bpm, max_bpm = self.bpm_mapper.get_bpm_range(emotion_label)
        logging.info(f"Recommended BPM range for '{emotion_label}' emotion: {min_bpm}-{max_bpm}")

        # 3. getsongbpm API를 통해 노래 검색 (또는 시뮬레이션 데이터 사용)
        recommended_songs = self.get_songs_by_bpm_range(min_bpm, max_bpm, limit=3) # <-- limit=3 유지
        
        if recommended_songs:
            logging.info("\n--- Recommended Music List ---")
            for i, song in enumerate(recommended_songs):
                logging.info(f"{i+1}. Title: {song['title']}, Artist: {song['artist']}, BPM: {song['bpm']}, Cover: {song.get('album_cover_url', 'N/A')}")
        else:
            logging.info("\nSorry, no music found for the current BPM range.")
            logging.info("Please try another emotion or try again later.")
            
        return recommended_songs

# MockMusicRecommender 클래스를 MusicRecommender 클래스 외부로 이동
class MockMusicRecommender(MusicRecommender):
    """
    API 키가 없거나 개발 시에 사용할 모의(Mock) 추천기입니다.
    실제 API 호출 없이 미리 정의된 데이터를 반환합니다.
    """
    def __init__(self, getsongbpm_api_key: str = None):
        super().__init__(getsongbpm_api_key) 
        logging.info("--- Mock Music Recommender initialized (using mock data only) ---")

    def get_songs_by_bpm_range(self, min_bpm: int, max_bpm: int, limit: int = 5): # Mock 클래스도 limit 파라미터 받도록
        logging.info(f"Mock API call: Simulating search for songs in BPM {min_bpm}~{max_bpm} range with limit {limit}...")
        time.sleep(1) # 모의 지연
        
        # Mock 데이터셋을 더 다양하게 구성하고 "(Mock)" 접미사 추가
        mock_data = [
            {"title": "Mock Pop Song (Upbeat)", "artist": "Mock Artist", "bpm": 120, "album_cover_url": "https://placehold.co/140x140/A3C8F5/000000?text=MockPop"},
            {"title": "Mock Jazz Tune (Relaxed)", "artist": "Jazz Cat", "bpm": 90, "album_cover_url": "https://placehold.co/140x140/4A90E2/FFFFFF?text=MockJazz"},
            {"title": "Mock Rock Anthem (Energetic)", "artist": "Rock Band", "bpm": 150, "album_cover_url": "https://placehold.co/140x140/333333/FFFFFF?text=MockRock"},
            {"title": "Mock Chill Vibes (Calm)", "artist": "Lo-Fi Beats", "bpm": 70, "album_cover_url": "https://placehold.co/140x140/6C7A89/FFFFFF?text=MockChill"},
            {"title": "Mock Upbeat Track (Happy)", "artist": "Energetic Duo", "bpm": 135, "album_cover_url": "https://placehold.co/140x140/FF6B6B/FFFFFF?text=MockUpbeat"},
            {"title": "Mock Summer Hit (Joyful)", "artist": "Sunshine Band", "bpm": 128, "album_cover_url": "https://placehold.co/140x140/FFD700/000000?text=MockSummer"},
            {"title": "Mock Rainy Day (Sad)", "artist": "Blue Mood", "bpm": 65, "album_cover_url": "https://placehold.co/140x140/87CEEB/000000?text=MockRain"},
            {"title": "Mock Workout Jam (Intense)", "artist": "Fitness Crew", "bpm": 140, "album_cover_url": "https://placehold.co/140x140/FF4500/FFFFFF?text=MockGym"},
        ]

        # 사용자 입력에 따라 다른 mock 데이터를 반환하는 간단한 로직 (예시)
        # 이 부분은 실제 API 호출이 아니므로, BPM 범위 필터링은 Mock 데이터 내에서만 이루어집니다.
        filtered_by_bpm_and_text = []
        for song in mock_data:
            if min_bpm <= song["bpm"] <= max_bpm:
                # '신나는' 감정일 경우 더 활기찬 Mock 데이터를 선호
                if "신나는" in self.sentiment_analyzer.analyze_sentiment("신나는 음악 추천해줘!")['label'] and song["bpm"] > 110:
                    filtered_by_bpm_and_text.append(song)
                # '조용한' 감정일 경우 더 차분한 Mock 데이터를 선호
                elif "조용한" in self.sentiment_analyzer.analyze_sentiment("조용한 음악 추천해줘!")['label'] and song["bpm"] < 90:
                    filtered_by_bpm_and_text.append(song)
                else:
                    filtered_by_bpm_and_text.append(song)

        if not filtered_by_bpm_and_text:
            logging.warning(f"No specific mock songs found for BPM range {min_bpm}~{max_bpm}. Returning random mock songs.")
            return random.sample(mock_data, min(limit, len(mock_data)))
            
        return random.sample(filtered_by_bpm_and_text, min(limit, len(filtered_by_bpm_and_text)))


# 이 파일이 직접 실행될 때만 실행되는 테스트 코드
if __name__ == "__main__":
    getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY", "YOUR_GETSONGBPM_API_KEY_HERE")

    if getsongbpm_api_key == "YOUR_GETSONGBPM_API_KEY_HERE":
        logging.warning("\n[WARNING]: getsongbpm API key is not set. Proceeding with mock data for testing.")
        recommender = MockMusicRecommender(getsongbpm_api_key) # Mock 클래스 사용
    else:
        logging.info("GETSONGBPM_API_KEY is set. Proceeding with actual API calls.")
        recommender = MusicRecommender(getsongbpm_api_key) # 실제 API 키로 MusicRecommender 사용

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
        recommender.recommend_music(user_text)
        logging.info("\n" + "="*70 + "\n")