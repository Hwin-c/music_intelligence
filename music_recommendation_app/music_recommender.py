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
# flask run 환경에서 상대 경로 임포트 문제 해결을 위해 sys.path 조작 방식을 다시 사용합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
nlp_dir = os.path.join(current_dir, 'natural-language-processing')

if nlp_dir not in sys.path:
    sys.path.insert(0, nlp_dir) # sys.path의 맨 앞에 추가하여 우선순위를 높임
    logging.debug(f"Added {nlp_dir} to sys.path for NLP modules.")

try:
    # 실제 SentimentAnalyzer 및 BPMMapper 임포트 시도 (직접 모듈 이름 사용)
    from sentiment_analyzer import SentimentAnalyzer
    from bpm_mapper import BPMMapper
    logging.debug("SentimentAnalyzer and BPMMapper imported successfully.")
except ImportError as e:
    logging.warning(f"Failed to import actual NLP modules: {e}. Using Mock versions.")
    # 실제 모듈 임포트 실패 시 Mock 버전 사용
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
            self.bpm_map = {
                "긍정": (110, 140),
                "부정": (60, 90),
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


class MusicRecommender:
    """
    사용자의 감정을 분석하여 BPM 기반으로 음악을 추천하는 클래스입니다.
    """
    def __init__(self, getsongbpm_api_key: str):
        """
        MusicRecommender를 초기화합니다.

        Args:
            getsongbpm_api_key (str): getsong.co API 키.
        """
        self.sentiment_analyzer = SentimentAnalyzer()
        self.bpm_mapper = BPMMapper()
        self.getsongbpm_api_key = getsongbpm_api_key
        self.getsongbpm_base_url = "https://api.getsong.co/" 
        logging.info(f"Music recommendation system initialized with API Base URL: {self.getsongbpm_base_url}")

    def _call_getsongbpm_api(self, endpoint: str, params: dict = None):
        """
        getsong.co API를 호출하는 내부 도우미 메서드입니다.
        API Key는 URL 파라미터로 전송합니다.
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
            
            # API 응답 본문을 로깅하여 실제 데이터를 확인
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
        return None

    def get_songs_by_bpm_range(self, min_bpm: int, max_bpm: int, limit: int = 5, emotion_label: str = "neutral"):
        """
        getsong.co API를 사용하여 특정 BPM 범위 내의 노래를 검색합니다.
        실제 API 호출을 시도하고, 실패 시 Mock 데이터로 대체합니다.
        """
        logging.info(f"Attempting to search for songs in BPM range {min_bpm}~{max_bpm} using getsong.co API...")
        
        found_songs = []
        seen_titles = set() # 중복 제목을 추적하기 위한 셋
        api_call_successful = False 

        # 감정 기반 검색 키워드 매핑
        emotion_keywords = {
            "긍정": ["happy", "upbeat", "energetic", "joyful", "party", "celebration"],
            "부정": ["sad", "melancholy", "downbeat", "gloomy", "depressed"],
            "공부": ["focus", "study", "concentration", "ambient", "instrumental"],
            "화가 나": ["angry", "rage", "intense", "aggressive"],
            "스트레스": ["relax", "calm", "chill", "soothing"],
            "편안": ["relax", "calm", "chill", "peaceful"],
            "불안": ["soothing", "calm", "meditation", "peaceful"],
            "neutral": ["easy listening", "background", "chill"]
        }
        
        # 기본 장르 키워드 (fallback 및 추가 다양성)
        genre_keywords = ["pop", "dance", "rock", "electronic", "jazz", "hip hop", "ballad", "r&b", "k-pop", "indie", "soul", "funk", "classical"]

        # 감정 기반 키워드를 우선 사용하고, 장르 키워드를 추가합니다.
        # 중복을 피하기 위해 set을 사용합니다.
        combined_queries = set(emotion_keywords.get(emotion_label, emotion_keywords["neutral"]))
        combined_queries.update(genre_keywords)
        
        # 검색 쿼리 순서를 무작위로 섞어서 다양한 결과를 시도
        search_queries_list = list(combined_queries)
        random.shuffle(search_queries_list)

        try:
            for query in search_queries_list:
                params = {"type": "song", "lookup": query, "limit": 20} 
                api_response = self._call_getsongbpm_api("search/", params) 

                if api_response and api_response.get("search"): 
                    api_call_successful = True 
                    for song_data in api_response["search"]: 
                        song_title = song_data.get("title", "Unknown Title")
                        song_uri = song_data.get("uri", "#") 
                        
                        # 아티스트 이름 추출
                        artist_name = "Unknown Artist"
                        artist_info = song_data.get("artist") 
                        if isinstance(artist_info, dict): 
                            artist_name = artist_info.get("name", "Unknown Artist")
                        
                        # 장르 추출
                        genres = []
                        if isinstance(artist_info, dict):
                            artist_genres = artist_info.get("genres")
                            if isinstance(artist_genres, list):
                                genres = artist_genres
                        
                        # 노래 제목이 장르 이름과 동일한지 확인하여 필터링
                        # 예를 들어, 제목이 "Pop"이고 장르에 "pop"이 있다면 제외
                        is_title_a_genre = False
                        if song_title.lower() in [g.lower() for g in genres]:
                            is_title_a_genre = True
                        
                        # 이미 추가된 제목이거나 장르 이름과 동일한 제목이면 건너뛰기
                        if song_title in seen_titles or is_title_a_genre:
                            continue 

                        song_bpm = song_data.get("tempo") 

                        if song_bpm is not None:
                            try:
                                song_bpm = int(song_bpm) 
                                if min_bpm <= song_bpm <= max_bpm:
                                    found_songs.append({
                                        "title": song_title,
                                        "artist": artist_name,
                                        "bpm": song_bpm,
                                        "uri": song_uri, 
                                        "genres": genres 
                                    })
                                    seen_titles.add(song_title) # 제목을 셋에 추가
                                    if len(found_songs) >= limit: 
                                        break
                            except ValueError:
                                logging.warning(f"Invalid BPM value received for song {song_title}: {song_bpm}")
                if len(found_songs) >= limit:
                    break

        except Exception as e:
            logging.error(f"Error during getsong.co API search: {e}")
            logging.error(traceback.format_exc())
            api_call_successful = False 
            
        if not api_call_successful or not found_songs:
            logging.warning(f"API call failed or no relevant songs found from getsong.co API for BPM range {min_bpm}~{max_bpm}. Falling back to mock data.")
            mock_songs_data = [
                {"title": "Dancing Monkey (Mock)", "artist": "Tones And I (Mock)", "bpm": 98, "uri": "#", "genres": ["Pop", "Indie"]},
                {"title": "Shape of You (Mock)", "artist": "Ed Sheeran (Mock)", "bpm": 96, "uri": "#", "genres": ["Pop", "R&B"]},
                {"title": "Blinding Lights (Mock)", "artist": "The Weeknd (Mock)", "bpm": 171, "uri": "#", "genres": ["Pop", "Synth-pop"]},
                {"title": "Dynamite (Mock)", "artist": "BTS (Mock)", "bpm": 114, "uri": "#", "genres": ["K-Pop", "Disco-Pop"]},
                {"title": "Bad Guy (Mock)", "artist": "Billie Eilish (Mock)", "bpm": 135, "uri": "#", "genres": ["Pop", "Electropop"]},
                {"title": "Old Town Road (Mock)", "artist": "Lil Nas X (Mock)", "bpm": 136, "uri": "#", "genres": ["Country Rap"]},
                {"title": "Someone You Loved (Mock)", "artist": "Lewis Capaldi (Mock)", "bpm": 109, "uri": "#", "genres": ["Pop", "Ballad"]},
                {"title": "Dance Monkey (Mock)", "artist": "Tones And I (Mock)", "bpm": 98, "uri": "#", "genres": ["Pop", "Indie"]}, # 중복 테스트용
                {"title": "Happy (Mock)", "artist": "Pharrell Williams (Mock)", "bpm": 160, "uri": "#", "genres": ["Pop", "Soul"]},
                {"title": "Uptown Funk (Mock)", "artist": "Mark Ronson (Mock)", "bpm": 115, "uri": "#", "genres": ["Funk", "Pop"]},
                {"title": "Bohemian Rhapsody (Mock)", "artist": "Queen (Mock)", "bpm": 144, "uri": "#", "genres": ["Rock", "Classic Rock"]},
            ]
            # Mock 데이터에서도 중복 제목과 장르 이름과 동일한 제목을 제거하도록 필터링
            unique_mock_songs = []
            mock_seen_titles = set()
            for song in mock_songs_data:
                is_mock_title_a_genre = False
                if song["title"].lower() in [g.lower() for g in song.get("genres", [])]:
                    is_mock_title_a_genre = True

                if song["title"] not in mock_seen_titles and not is_mock_title_a_genre:
                    unique_mock_songs.append(song)
                    mock_seen_titles.add(song["title"])
            return random.sample(unique_mock_songs, min(limit, len(unique_mock_songs)))
        
        # 실제 API 결과에서도 중복 제거 후 샘플링
        unique_found_songs = []
        found_seen_titles = set()
        for song in found_songs:
            if song["title"] not in found_seen_titles:
                unique_found_songs.append(song)
                found_seen_titles.add(song["title"])
        return random.sample(unique_found_songs, min(limit, len(unique_found_songs)))

    def recommend_music(self, user_text: str):
        """
        사용자 텍스트를 기반으로 음악을 추천합니다.

        Args:
            user_text (str): 사용자의 감정 표현 텍스트.

        Returns:
            list: 추천된 음악 리스트 (제목, 아티스트, BPM, URI, 장르 포함).
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

        # 3. getsong.co API를 통해 노래 검색 (또는 시뮬레이션 데이터 사용)
        # emotion_label을 get_songs_by_bpm_range로 전달하여 감정 기반 검색 키워드에 활용
        recommended_songs = self.get_songs_by_bpm_range(min_bpm, max_bpm, limit=3, emotion_label=emotion_label) 
        
        if recommended_songs:
            logging.info("\n--- Recommended Music List ---")
            for i, song in enumerate(recommended_songs):
                # 로깅 메시지에 URI와 장르 추가
                genres_str = ", ".join(song.get('genres', [])) if song.get('genres') else "N/A"
                logging.info(f"{i+1}. Title: {song['title']}, Artist: {song['artist']}, BPM: {song['bpm']}, Genres: {genres_str}, URI: {song['uri']}")
        else:
            logging.info("\nSorry, no music found for the current BPM range.")
            logging.info("Please try another emotion or try again later.")
            
        return recommended_songs


class MockMusicRecommender(MusicRecommender):
    """
    API 키가 없거나 개발 시에 사용할 모의(Mock) 추천기입니다.
    실제 API 호출 없이 미리 정의된 데이터를 반환합니다.
    """
    def __init__(self, getsongbpm_api_key: str = None):
        super().__init__(getsongbpm_api_key) 
        logging.info("--- Mock Music Recommender initialized (using mock data only) ---")

    def get_songs_by_bpm_range(self, min_bpm: int, max_bpm: int, limit: int = 5, emotion_label: str = "neutral"): # emotion_label 파라미터 추가
        logging.info(f"Mock API call: Simulating search for songs in BPM {min_bpm}~{max_bpm} range with limit {limit}...")
        time.sleep(1) # 모의 지연
        
        # Pylance 경고 해결: 변수 초기화
        filtered_by_bpm_and_text = [] 

        # Mock 데이터셋을 더 다양하게 구성하고 "(Mock)" 접미사 추가
        mock_data = [
            {"title": "Dancing Monkey (Mock)", "artist": "Tones And I (Mock)", "bpm": 98, "uri": "#", "genres": ["Pop", "Indie"]},
            {"title": "Shape of You (Mock)", "artist": "Ed Sheeran (Mock)", "bpm": 96, "uri": "#", "genres": ["Pop", "R&B"]},
            {"title": "Blinding Lights (Mock)", "artist": "The Weeknd (Mock)", "bpm": 171, "uri": "#", "genres": ["Pop", "Synth-pop"]},
            {"title": "Dynamite (Mock)", "artist": "BTS (Mock)", "bpm": 114, "uri": "#", "genres": ["K-Pop", "Disco-Pop"]},
            {"title": "Bad Guy (Mock)", "artist": "Billie Eilish (Mock)", "bpm": 135, "uri": "#", "genres": ["Pop", "Electropop"]},
            {"title": "Old Town Road (Mock)", "artist": "Lil Nas X (Mock)", "bpm": 136, "uri": "#", "genres": ["Country Rap"]},
            {"title": "Someone You Loved (Mock)", "artist": "Lewis Capaldi (Mock)", "bpm": 109, "uri": "#", "genres": ["Pop", "Ballad"]},
            {"title": "Happy (Mock)", "artist": "Pharrell Williams (Mock)", "bpm": 160, "uri": "#", "genres": ["Pop", "Soul"]},
            {"title": "Uptown Funk (Mock)", "artist": "Mark Ronson (Mock)", "bpm": 115, "uri": "#", "genres": ["Funk", "Pop"]},
            {"title": "Bohemian Rhapsody (Mock)", "artist": "Queen (Mock)", "bpm": 144, "uri": "#", "genres": ["Rock", "Classic Rock"]},
            {"title": "Lose Yourself (Mock)", "artist": "Eminem (Mock)", "bpm": 171, "uri": "#", "genres": ["Hip Hop", "Rap"]},
            {"title": "Imagine (Mock)", "artist": "John Lennon (Mock)", "bpm": 75, "uri": "#", "genres": ["Pop", "Soft Rock"]},
            {"title": "What a Wonderful World (Mock)", "artist": "Louis Armstrong (Mock)", "bpm": 82, "uri": "#", "genres": ["Jazz", "Vocal"]},
            {"title": "Stairway to Heaven (Mock)", "artist": "Led Zeppelin (Mock)", "bpm": 82, "uri": "#", "genres": ["Rock", "Hard Rock"]},
            {"title": "Hotel California (Mock)", "artist": "Eagles (Mock)", "bpm": 147, "uri": "#", "genres": ["Rock", "Classic Rock"]},
            {"title": "Yesterday (Mock)", "artist": "The Beatles (Mock)", "bpm": 94, "uri": "#", "genres": ["Pop", "Rock"]},
            {"title": "Smells Like Teen Spirit (Mock)", "artist": "Nirvana (Mock)", "bpm": 117, "uri": "#", "genres": ["Grunge", "Rock"]},
            {"title": "Billie Jean (Mock)", "artist": "Michael Jackson (Mock)", "bpm": 117, "uri": "#", "genres": ["Pop", "Funk"]},
            {"title": "Like a Rolling Stone (Mock)", "artist": "Bob Dylan (Mock)", "bpm": 95, "uri": "#", "genres": ["Folk Rock"]},
            {"title": "One (Mock)", "artist": "U2 (Mock)", "bpm": 91, "uri": "#", "genres": ["Rock"]},
        ]

        # 사용자 입력에 따라 다른 mock 데이터를 반환하는 간단한 로직 (예시)
        for song in mock_data:
            if min_bpm <= song["bpm"] <= max_bpm:
                # self.sentiment_analyzer가 MockSentimentAnalyzer이므로 직접 호출
                sentiment_label_for_mock = self.sentiment_analyzer.analyze_sentiment("긍정적인 음악 추천해줘!")['label']
                if "긍정" in sentiment_label_for_mock and song["bpm"] > 110:
                    filtered_by_bpm_and_text.append(song)
                elif "부정" in sentiment_label_for_mock and song["bpm"] < 90:
                    filtered_by_bpm_and_text.append(song)
                else:
                    filtered_by_bpm_and_text.append(song)

        if not filtered_by_bpm_and_text:
            logging.warning(f"No specific mock songs found for BPM range {min_bpm}~{max_bpm}. Returning random mock songs.")
            # Mock 데이터에서도 중복 제목과 장르 이름과 동일한 제목을 제거하도록 필터링
            unique_mock_songs = []
            mock_seen_titles = set()
            for song in mock_data:
                is_mock_title_a_genre = False
                if song["title"].lower() in [g.lower() for g in song.get("genres", [])]:
                    is_mock_title_a_genre = True

                if song["title"] not in mock_seen_titles and not is_mock_title_a_genre:
                    unique_mock_songs.append(song)
                    mock_seen_titles.add(song["title"])
            return random.sample(unique_mock_songs, min(limit, len(unique_mock_songs)))
            
        # filtered_by_bpm_and_text가 비어있지 않은 경우, 해당 리스트에서 샘플링하여 반환
        unique_filtered_songs = []
        filtered_seen_titles = set()
        for song in filtered_by_bpm_and_text:
            if song["title"] not in filtered_seen_titles:
                unique_filtered_songs.append(song)
                filtered_seen_titles.add(song["title"])
        return random.sample(unique_filtered_songs, min(limit, len(unique_filtered_songs)))