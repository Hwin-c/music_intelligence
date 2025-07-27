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
            # BPM 외에 danceability, acousticness 범위 추가 (0-100)
            self.emotion_features_map = {
                "긍정": {"bpm": (110, 140), "danceability": (70, 100), "acousticness": (0, 30)}, # 신나는 음악은 댄서블하고 어쿠스틱하지 않음
                "부정": {"bpm": (60, 90), "danceability": (0, 40), "acousticness": (50, 100)}, # 슬픈 음악은 덜 댄서블하고 어쿠스틱할 수 있음
                "공부": {"bpm": (80, 110), "danceability": (0, 50), "acousticness": (30, 70)}, # 집중 음악은 차분하고 적당히 어쿠스틱
                "화가 나": {"bpm": (130, 180), "danceability": (50, 90), "acousticness": (0, 20)}, # 격렬한 음악은 빠르고 댄서블하며 어쿠스틱하지 않음
                "스트레스": {"bpm": (70, 100), "danceability": (20, 60), "acousticness": (40, 90)}, # 휴식 음악은 느리고 덜 댄서블하며 어쿠스틱할 수 있음
                "편안": {"bpm": (70, 100), "danceability": (20, 60), "acousticness": (40, 90)},
                "불안": {"bpm": (95, 125), "danceability": (30, 70), "acousticness": (20, 60)},
                "neutral": {"bpm": (90, 120), "danceability": (40, 80), "acousticness": (20, 70)}
            }
            logging.info("BPMMapper (with audio features) initialized.")

        # get_audio_feature_ranges 메서드 추가
        def get_audio_feature_ranges(self, emotion_label: str):
            logging.debug(f"BPMMapper: Mapping audio features for '{emotion_label}'")
            return self.emotion_features_map.get(emotion_label, self.emotion_features_map["neutral"])


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
        self.sentiment_analyzer = SentimentAnalyzer()
        self.bpm_mapper = BPMMapper() # BPMMapper는 이제 오디오 특성도 반환
        self.getsongbpm_api_key = getsongbpm_api_key
        # API 엔드포인트 변경: /audio-features 사용
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

    def get_songs_by_keywords_and_bpm(self, emotion_label: str, min_bpm: int, max_bpm: int, limit: int = 5):
        """
        getsong.co API를 사용하여 키워드 및 BPM 범위 내의 노래를 검색합니다.
        '/search' 엔드포인트를 활용하여 더 다양한 제목을 얻도록 시도합니다.
        """
        logging.info(f"Attempting to search for songs using keywords and BPM range {min_bpm}~{max_bpm} for '{emotion_label}' emotion via getsong.co API...")
        
        found_songs = []
        seen_titles = set() # 중복 제목을 추적하기 위한 셋
        api_call_successful = False 

        # 감정 기반 검색 키워드 매핑 (더 다양하고 구체적인 키워드 추가)
        emotion_keywords = {
            "긍정": ["happy", "upbeat", "energetic", "joyful", "party", "celebration", "optimistic", "bright", "good vibes", "dancing", "fun"],
            "부정": ["sad", "melancholy", "downbeat", "gloomy", "depressed", "lonely", "heartbreak", "somber", "blue"],
            "공부": ["focus", "study", "concentration", "ambient", "instrumental", "classical", "lo-fi", "calm", "relaxing"],
            "화가 나": ["angry", "rage", "intense", "aggressive", "metal", "punk", "hard rock", "rebellion"],
            "스트레스": ["relax", "calm", "chill", "soothing", "meditation", "peaceful", "unwind"],
            "편안": ["relax", "calm", "chill", "peaceful", "smooth", "mellow", "serene"],
            "불안": ["soothing", "calm", "meditation", "peaceful", "gentle", "comforting"],
            "neutral": ["easy listening", "background music", "chill out", "acoustic", "mellow"]
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
                params = {"type": "song", "lookup": query, "limit": 20} # limit 늘려서 더 많은 결과 시도
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
                        # 예: 제목이 "Pop"이고 장르에 "pop"이 있다면 제외
                        is_title_a_genre = False
                        for g in genres:
                            if song_title.lower() == g.lower():
                                is_title_a_genre = True
                                break
                        
                        # 이미 추가된 제목이거나 장르 이름과 동일한 제목이면 건너뛰기
                        if song_title in seen_titles or is_title_a_genre:
                            continue 

                        song_bpm = song_data.get("tempo") 

                        if song_bpm is not None:
                            try:
                                song_bpm = int(song_bpm) 
                                if min_bpm <= song_bpm <= max_bpm: # 클라이언트 측 BPM 필터링
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
                    break # 현재 쿼리에서 충분한 노래를 찾았으면 다음 쿼리로 넘어가지 않음

        except Exception as e:
            logging.error(f"Error during getsong.co API search: {e}")
            logging.error(traceback.format_exc())
            api_call_successful = False 
            
        if not api_call_successful or not found_songs:
            logging.warning(f"API call failed or no relevant songs found from getsong.co API. Falling back to mock data.")
            mock_songs_data = [
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

        # 2. 감정에 따른 BPM 범위 매핑 (BPMMapper에서 오디오 특성 범위 가져오기)
        feature_ranges = self.bpm_mapper.get_audio_feature_ranges(emotion_label)
        min_bpm, max_bpm = feature_ranges["bpm"]
        
        logging.info(f"Recommended BPM range for '{emotion_label}' emotion: {min_bpm}-{max_bpm}")

        # 3. getsong.co API를 통해 노래 검색 (이제 /search/ 엔드포인트만 사용)
        recommended_songs = self.get_songs_by_keywords_and_bpm(emotion_label, min_bpm, max_bpm, limit=3) 
        
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

    def get_songs_by_keywords_and_bpm(self, emotion_label: str, min_bpm: int, max_bpm: int, limit: int = 5): # emotion_label 파라미터 추가
        logging.info(f"Mock API call: Simulating search for songs with keywords and BPM range {min_bpm}~{max_bpm} for '{emotion_label}' emotion with limit {limit}...")
        time.sleep(1) # 모의 지연
        
        # Pylance 경고 해결: 변수 초기화
        filtered_by_features = [] 

        # Mock 데이터셋을 더 다양하게 구성하고 "(Mock)" 접미사 추가
        # 이제 Mock 데이터도 더 다양한 제목을 가집니다.
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

        # 감정 레이블에 따라 Mock 데이터 필터링 (간단한 예시)
        feature_ranges = self.bpm_mapper.get_audio_feature_ranges(emotion_label)
        min_bpm, max_bpm = feature_ranges["bpm"]
        # Mock 데이터에는 danceability, acousticness가 없으므로 BPM만으로 필터링
        
        for song in mock_data:
            if min_bpm <= song["bpm"] <= max_bpm:
                filtered_by_features.append(song)

        if not filtered_by_features:
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
            
        # filtered_by_features에서 중복 제거 후 샘플링
        unique_filtered_songs = []
        filtered_seen_titles = set()
        for song in filtered_by_features:
            is_title_a_genre = False
            if song["title"].lower() in [g.lower() for g in song.get("genres", [])]:
                is_title_a_genre = True
            
            if song["title"] not in filtered_seen_titles and not is_title_a_genre:
                unique_filtered_songs.append(song)
                filtered_seen_titles.add(song["title"])
        
        return random.sample(unique_filtered_songs, min(limit, len(unique_filtered_songs)))


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