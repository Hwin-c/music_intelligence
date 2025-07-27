import requests
import random
import os
import sys
import logging
import time

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock SentimentAnalyzer 및 BPMMapper 클래스 (실제 모듈이 없을 경우를 대비)
# 이 클래스들은 natural-language-processing 디렉토리에 실제 파일로 존재해야 합니다.
# 만약 해당 파일들이 없다면, 이 코드를 natural-language-processing/sentiment_analyzer.py
# 및 natural-language-processing/bpm_mapper.py 에 각각 넣어주셔야 합니다.
class MockSentimentAnalyzer:
    def analyze_sentiment(self, text: str):
        logging.debug(f"MockSentimentAnalyzer: Analyzing '{text}'")
        # 간단한 텍스트 기반 감정 매핑 (예시)
        if "신나" in text or "기분 좋" in text or "활기찬" in text:
            return {"label": "happy", "score": 0.9}
        elif "슬프" in text or "우울" in text:
            return {"label": "sad", "score": 0.8}
        elif "공부" in text or "집중" in text or "잔잔" in text:
            return {"label": "calm", "score": 0.7}
        elif "화가 나" in text or "분노" in text:
            return {"label": "angry", "score": 0.85}
        elif "스트레스" in text or "쉬고 싶" in text:
            return {"label": "relaxed", "score": 0.75}
        else:
            return {"label": "neutral", "score": 0.5}

class MockBPMMapper:
    def __init__(self):
        self.bpm_map = {
            "happy": (120, 160),
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
            logging.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
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
        
        **주의:** getsongbpm API는 직접적인 'BPM 범위 검색' 기능을 명시적으로 제공하지 않을 수 있습니다.
        이 함수는 해당 API가 트랙 검색 결과를 반환하고, 그 결과에 BPM 정보와 앨범 커버 URL이 포함되어 있음을
        가정하여 클라이언트 측에서 BPM으로 필터링하는 방식으로 구현됩니다.
        실제 API 문서를 확인하여 가장 효율적인 검색 방식과 앨범 커버 URL 필드명을 적용해야 합니다.
        (예: 장르 또는 인기곡 검색 후 BPM 필터링 및 앨범 커버 URL 추출)
        """
        logging.info(f"Attempting to search for songs in BPM range {min_bpm}~{max_bpm} from getsongbpm API...")
        
        logging.warning("Direct BPM range search on getsongbpm API might be limited.")
        logging.info("Simulating with mock data that includes album cover URLs.")
        
        # 모의 데이터에 앨범 커버 URL 추가
        mock_songs_data = [
            {"title": "기분 좋은 아침", "artist": "김미소", "bpm": 130, "album_cover_url": "https://placehold.co/140x140/FFD700/000000?text=Happy"},
            {"title": "고요한 숲길", "artist": "이평화", "bpm": 75, "album_cover_url": "https://placehold.co/140x140/ADD8E6/000000?text=Calm"},
            {"title": "집중의 순간", "artist": "박몰입", "bpm": 100, "album_cover_url": "https://placehold.co/140x140/90EE90/000000?text=Focus"},
            {"title": "파워 업!", "artist": "최에너지", "bpm": 155, "album_cover_url": "https://placehold.co/140x140/FF4500/FFFFFF?text=Power"},
            {"title": "차분한 저녁", "artist": "정고요", "bpm": 60, "album_cover_url": "https://placehold.co/140x140/8A2BE2/FFFFFF?text=Evening"},
            {"title": "활기찬 하루", "artist": "강다이나믹", "bpm": 120, "album_cover_url": "https://placehold.co/140x140/00CED1/FFFFFF?text=Dynamic"},
            {"title": "생각의 흐름", "artist": "윤명상", "bpm": 90, "album_cover_url": "https://placehold.co/140x140/DDA0DD/000000?text=Thought"},
            {"title": "분노의 질주", "artist": "서파워", "bpm": 140, "album_cover_url": "https://placehold.co/140x140/B22222/FFFFFF?text=Rage"},
            {"title": "슬픈 빗소리", "artist": "오감성", "bpm": 70, "album_cover_url": "https://placehold.co/140x140/6A5ACD/FFFFFF?text=Sad"},
            {"title": "새로운 도전", "artist": "조열정", "bpm": 115, "album_cover_url": "https://placehold.co/140x140/FF8C00/000000?text=Challenge"},
        ]
        
        filtered_songs = [
            song for song in mock_songs_data 
            if min_bpm <= song["bpm"] <= max_bpm
        ]
        
        if not filtered_songs:
            logging.warning(f"No songs found in mock data for BPM range {min_bpm}~{max_bpm}. Returning random mock songs.")
            return random.sample(mock_songs_data, min(limit, len(mock_songs_data)))
            
        return random.sample(filtered_songs, min(limit, len(filtered_songs)))


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
        # 이 함수가 앨범 커버 URL을 포함한 데이터를 반환하도록 수정되었습니다.
        recommended_songs = self.get_songs_by_bpm_range(min_bpm, max_bpm, limit=5)
        
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
    def __init__(self, getsongbpm_api_key: str = None): # api_key를 선택적으로 받도록 수정
        # MockMusicRecommender는 실제 API 키가 필요 없으므로, getsongbpm_api_key를 None으로 설정
        super().__init__(api_key) 
        logging.info("--- Mock Music Recommender initialized ---")

    def get_songs_by_bpm_range(self, min_bpm: int, max_bpm: int, limit: int = 5):
        logging.info(f"Mock API call: Simulating search for songs in BPM {min_bpm}~{max_bpm} range...")
        
        mock_songs_data = [
            {"title": "기분 좋은 아침", "artist": "김미소", "bpm": 130, "album_cover_url": "https://placehold.co/140x140/FFD700/000000?text=Happy"},
            {"title": "고요한 숲길", "artist": "이평화", "bpm": 75, "album_cover_url": "https://placehold.co/140x140/ADD8E6/000000?text=Calm"},
            {"title": "집중의 순간", "artist": "박몰입", "bpm": 100, "album_cover_url": "https://placehold.co/140x140/90EE90/000000?text=Focus"},
            {"title": "파워 업!", "artist": "최에너지", "bpm": 155, "album_cover_url": "https://placehold.co/140x140/FF4500/FFFFFF?text=Power"},
            {"title": "차분한 저녁", "artist": "정고요", "bpm": 60, "album_cover_url": "https://placehold.co/140x140/8A2BE2/FFFFFF?text=Evening"},
            {"title": "활기찬 하루", "artist": "강다이나믹", "bpm": 120, "album_cover_url": "https://placehold.co/140x140/00CED1/FFFFFF?text=Dynamic"},
            {"title": "생각의 흐름", "artist": "윤명상", "bpm": 90, "album_cover_url": "https://placehold.co/140x140/DDA0DD/000000?text=Thought"},
            {"title": "분노의 질주", "artist": "서파워", "bpm": 140, "album_cover_url": "https://placehold.co/140x140/B22222/FFFFFF?text=Rage"},
            {"title": "슬픈 빗소리", "artist": "오감성", "bpm": 70, "album_cover_url": "https://placehold.co/140x140/6A5ACD/FFFFFF?text=Sad"},
            {"title": "새로운 도전", "artist": "조열정", "bpm": 115, "album_cover_url": "https://placehold.co/140x140/FF8C00/000000?text=Challenge"},
        ]
        
        filtered_songs = [
            song for song in mock_songs_data 
            if min_bpm <= song["bpm"] <= max_bpm
        ]
        
        if not filtered_songs:
            logging.warning(f"No songs found in mock data for BPM range {min_bpm}~{max_bpm}. Returning random mock songs.")
            return random.sample(mock_songs_data, min(limit, len(mock_songs_data)))
            
        return random.sample(filtered_songs, min(limit, len(filtered_songs)))

    def recommend_music(self, user_text: str):
        logging.debug(f"Mock 데이터로 음악 추천 요청: '{user_text}'")
        time.sleep(1) # 모의 지연
        
        mock_data = [
            {"title": "Mock Pop Song", "artist": "Mock Artist", "bpm": "120", "album_cover_url": "https://placehold.co/140x140/A3C8F5/000000?text=Pop"},
            {"title": "Mock Jazz Tune", "artist": "Jazz Cat", "bpm": "90", "album_cover_url": "https://placehold.co/140x140/4A90E2/FFFFFF?text=Jazz"},
            {"title": "Mock Rock Anthem", "artist": "Rock Band", "bpm": "150", "album_cover_url": "https://placehold.co/140x140/333333/FFFFFF?text=Rock"},
            {"title": "Mock Chill Vibes", "artist": "Lo-Fi Beats", "bpm": "70", "album_cover_url": "https://placehold.co/140x140/6C7A89/FFFFFF?text=Chill"},
            {"title": "Mock Upbeat Track", "artist": "Energetic Duo", "bpm": "135", "album_cover_url": "https://placehold.co/140x140/FF6B6B/FFFFFF?text=Upbeat"},
        ]

        if "신나는" in user_text:
             return [
                {"title": "Mock 신나는 곡 1", "artist": "신나는 아티스트", "bpm": "130", "album_cover_url": "https://placehold.co/140x140/FFD700/000000?text=Exciting1"},
                {"title": "Mock 신나는 곡 2", "artist": "신나는 그룹", "bpm": "145", "album_cover_url": "https://placehold.co/140x140/FFA500/000000?text=Exciting2"},
                {"title": "Mock 신나는 곡 3", "artist": "신나는 밴드", "bpm": "125", "album_cover_url": "https://placehold.co/140x140/FF4500/FFFFFF?text=Exciting3"},
            ]
        elif "조용한" in user_text or "공부" in user_text:
            return [
                {"title": "Mock 조용한 곡 1", "artist": "조용한 아티스트", "bpm": "60", "album_cover_url": "https://placehold.co/140x140/ADD8E6/000000?text=Quiet1"},
                {"title": "Mock 조용한 곡 2", "artist": "조용한 그룹", "bpm": "75", "album_cover_url": "https://placehold.co/140x140/87CEEB/000000?text=Quiet2"},
                {"title": "Mock 조용한 곡 3", "artist": "조용한 밴드", "bpm": "80", "album_cover_url": "https://placehold.co/140x140/6495ED/FFFFFF?text=Quiet3"},
            ]
        else:
            return random.sample(mock_data, 3) # 3개 랜덤 선택


# 이 파일이 직접 실행될 때만 실행되는 테스트 코드
if __name__ == "__main__":
    getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY", "YOUR_GETSONGBPM_API_KEY_HERE")

    if getsongbpm_api_key == "YOUR_GETSONGBPM_API_KEY_HERE":
        logging.warning("\n[WARNING]: getsongbpm API key is not set.")
        logging.info("Proceeding with mock data for testing instead of actual API calls.")
        recommender = MockMusicRecommender(getsongbpm_api_key) # Mock 클래스 사용
    else:
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