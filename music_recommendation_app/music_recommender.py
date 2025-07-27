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
            logging.info("MockBPMMapper initialized.")

        # get_audio_feature_ranges 메서드 추가
        def get_audio_feature_ranges(self, emotion_label: str):
            logging.debug(f"MockBPMMapper: Mapping audio features for '{emotion_label}'")
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

    def _call_getsongbpm_api(self, endpoint: str, params: dict = None, delay_seconds: int = 1):
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
        finally:
            # API 호출 후 지연 추가
            time.sleep(delay_seconds)
        return None

    def _calculate_relevance_score(self, song_data: dict, target_features: dict) -> float:
        """
        노래의 오디오 특성이 목표 범위에 얼마나 잘 부합하는지 점수를 계산합니다.
        점수는 0에서 3 사이의 값을 가집니다 (각 특성별 0~1점).
        """
        score = 0.0

        # BPM 점수
        song_bpm = song_data.get("tempo")
        min_bpm, max_bpm = target_features["bpm"]
        if song_bpm is not None:
            try:
                song_bpm = int(song_bpm)
                if min_bpm <= song_bpm <= max_bpm:
                    # 범위 중앙에 가까울수록 높은 점수 (0.5 ~ 1.0)
                    center_bpm = (min_bpm + max_bpm) / 2
                    range_half = (max_bpm - min_bpm) / 2
                    if range_half > 0:
                        score += (1 - abs(song_bpm - center_bpm) / range_half) * 0.5 + 0.5
                    else: # 범위가 0일 경우 (min=max)
                        score += 1.0 if song_bpm == min_bpm else 0.0
                else:
                    score += 0.0 # 범위 밖이면 0점
            except ValueError:
                pass # BPM이 유효하지 않으면 점수 없음

        # Danceability 점수
        song_danceability = song_data.get("danceability")
        min_dance, max_dance = target_features["danceability"]
        if song_danceability is not None:
            try:
                song_danceability = int(song_danceability)
                if min_dance <= song_danceability <= max_dance:
                    center_dance = (min_dance + max_dance) / 2
                    range_half = (max_dance - min_dance) / 2
                    if range_half > 0:
                        score += (1 - abs(song_danceability - center_dance) / range_half) * 0.5 + 0.5
                    else:
                        score += 1.0 if song_danceability == min_dance else 0.0
                else:
                    score += 0.0
            except ValueError:
                pass

        # Acousticness 점수
        song_acousticness = song_data.get("acousticness")
        min_acoustic, max_acoustic = target_features["acousticness"]
        if song_acousticness is not None:
            try:
                song_acousticness = int(song_acousticness)
                if min_acoustic <= song_acousticness <= max_acoustic:
                    center_acoustic = (min_acoustic + max_acoustic) / 2
                    range_half = (max_acoustic - min_acoustic) / 2
                    if range_half > 0:
                        score += (1 - abs(song_acousticness - center_acoustic) / range_half) * 0.5 + 0.5
                    else:
                        score += 1.0 if song_acousticness == min_acoustic else 0.0
                else:
                    score += 0.0
            except ValueError:
                pass
        
        return score

    def get_ranked_songs_by_audio_features(self, emotion_label: str, limit: int = 3):
        """
        사용자의 감정에 따라 오디오 특성 기반으로 노래를 검색하고 랭킹을 매겨 반환합니다.
        '/search' 엔드포인트를 사용하여 더 다양한 제목을 얻도록 시도합니다.
        """
        logging.info(f"Attempting to search and rank songs based on audio features for '{emotion_label}' emotion via getsong.co API...")
        
        target_features = self.bpm_mapper.get_audio_feature_ranges(emotion_label)
        min_bpm, max_bpm = target_features["bpm"]

        candidate_songs = []
        seen_titles = set() # 중복 제목을 추적하기 위한 셋

        # 감정 기반 검색 키워드 매핑 (더 다양하고 구체적인 키워드 추가)
        # 이제 더 일반적인 음악 관련 키워드도 추가하여 검색 다양성을 높입니다.
        emotion_keywords_for_search = {
            "긍정": ["happy", "upbeat", "energetic", "joyful", "party", "celebration", "optimistic", "bright", "good vibes", "dancing", "fun", "pop", "dance"],
            "부정": ["sad", "melancholy", "downbeat", "gloomy", "depressed", "lonely", "heartbreak", "somber", "blue", "ballad", "acoustic"],
            "공부": ["focus", "study", "concentration", "ambient", "instrumental", "classical", "lo-fi", "calm", "relaxing", "jazz"],
            "화가 나": ["angry", "rage", "intense", "aggressive", "metal", "punk", "hard rock", "rebellion", "rock"],
            "스트레스": ["relax", "calm", "chill", "soothing", "meditation", "peaceful", "unwind", "ambient", "instrumental"],
            "편안": ["relax", "calm", "chill", "peaceful", "smooth", "mellow", "serene", "jazz", "soul"],
            "불안": ["soothing", "calm", "meditation", "peaceful", "gentle", "comforting", "ambient"],
            "neutral": ["easy listening", "background music", "chill out", "acoustic", "mellow", "pop", "soft rock"]
        }
        
        # 일반적인 음악 검색 키워드 (fallback 용도 또는 추가 다양성)
        general_music_keywords = ["song", "track", "music", "hit", "tune", "sound"]

        # 감정 기반 키워드를 우선 사용하고, 일반 음악 키워드를 추가합니다.
        # 중복을 피하기 위해 set을 사용합니다.
        combined_queries = set(emotion_keywords_for_search.get(emotion_label, emotion_keywords_for_search["neutral"]))
        combined_queries.update(general_music_keywords)
        
        # 검색 쿼리 순서를 무작위로 섞어서 다양한 결과를 시도
        search_queries_list = list(combined_queries)
        random.shuffle(search_queries_list)

        # 여러 쿼리를 통해 충분한 후보 노래를 수집
        max_api_calls = 5 # 최대 API 호출 횟수 제한 (rate limit 고려)
        calls_made = 0

        for query in search_queries_list:
            if len(candidate_songs) >= limit * 5 and calls_made >= max_api_calls: # 충분한 후보가 모였거나 최대 호출 횟수 도달 시 중단
                break

            params = {"type": "song", "lookup": query, "limit": 50} # 더 많은 결과를 가져와서 필터링할 여유를 줍니다.
            api_response = self._call_getsongbpm_api("search/", params, delay_seconds=1.5) # API 호출 간 지연 추가
            calls_made += 1

            if api_response and api_response.get("search"): 
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
                        # genres가 None일 경우 빈 리스트로 초기화하여 TypeError 방지
                        if isinstance(artist_genres, list):
                            genres = artist_genres
                        else:
                            genres = [] # None인 경우 빈 리스트로
                    else:
                        genres = [] # artist_info가 dict가 아닌 경우 빈 리스트로

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
                    song_danceability = song_data.get("danceability")
                    song_acousticness = song_data.get("acousticness")

                    # 모든 필수 오디오 특성 값이 유효한지 확인
                    if song_bpm is not None and song_danceability is not None and song_acousticness is not None:
                        try:
                            # 문자열로 된 숫자 값을 int로 변환
                            song_bpm = int(song_bpm)
                            song_danceability = int(song_danceability)
                            song_acousticness = int(song_acousticness)

                            # 관련성 점수 계산
                            relevance_score = self._calculate_relevance_score(
                                {"tempo": song_bpm, "danceability": song_danceability, "acousticness": song_acousticness},
                                target_features
                            )

                            if relevance_score > 0: # 점수가 0보다 큰 노래만 후보로 추가
                                candidate_songs.append({
                                    "title": song_title,
                                    "artist": artist_name,
                                    "bpm": song_bpm,
                                    "uri": song_uri, 
                                    "genres": genres,
                                    "relevance_score": relevance_score # 점수 저장
                                })
                                seen_titles.add(song_title) # 제목을 셋에 추가
                        except ValueError:
                            logging.warning(f"Invalid numeric value for audio feature in song {song_title}. Skipping.")
                    else:
                        logging.debug(f"Missing audio features for song {song_title}. Skipping for ranking.")
            
            if len(candidate_songs) >= limit * 5: # 충분한 후보가 모였으면 더 이상 검색하지 않음
                break

        # 후보 노래들을 관련성 점수 기준으로 내림차순 정렬
        candidate_songs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # 상위 N개 노래만 반환
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
                {"title": "Stairway to Heaven (Mock)", "artist": "Led Zeppelin (Mock)", "bpm": 82, "uri": "#", "genres": ["Rock", "Hard Rock"], "danceability": 25, "acousticness": 60},
                {"title": "Hotel California (Mock)", "artist": "Eagles (Mock)", "bpm": 147, "uri": "#", "genres": ["Rock", "Classic Rock"], "danceability": 50, "acousticness": 40},
                {"title": "Yesterday (Mock)", "artist": "The Beatles (Mock)", "bpm": 94, "uri": "#", "genres": ["Pop", "Rock"], "danceability": 45, "acousticness": 50},
                {"title": "Smells Like Teen Spirit (Mock)", "artist": "Nirvana (Mock)", "bpm": 117, "uri": "#", "genres": ["Grunge", "Rock"], "danceability": 60, "acousticness": 10},
                {"title": "Billie Jean (Mock)", "artist": "Michael Jackson (Mock)", "bpm": 117, "uri": "#", "genres": ["Pop", "Funk"], "danceability": 90, "acousticness": 5},
                {"title": "Like a Rolling Stone (Mock)", "artist": "Bob Dylan (Mock)", "bpm": 95, "uri": "#", "genres": ["Folk Rock"], "danceability": 40, "acousticness": 70},
                {"title": "One (Mock)", "artist": "U2 (Mock)", "bpm": 91, "uri": "#", "genres": ["Rock"], "danceability": 35, "acousticness": 60},
            ]
            # Mock 데이터에서도 중복 제목과 장르 이름과 동일한 제목을 제거하고 랭킹
            unique_mock_songs = []
            mock_seen_titles = set()
            for song in mock_songs_data:
                is_mock_title_a_genre = False
                if song["title"].lower() in [g.lower() for g in song.get("genres", [])]:
                    is_mock_title_a_genre = True

                if song["title"] not in mock_seen_titles and not is_mock_title_a_genre:
                    # Mock 데이터에도 relevance_score를 계산하여 추가
                    relevance_score = self._calculate_relevance_score(
                        {"tempo": song["bpm"], "danceability": song["danceability"], "acousticness": song["acousticness"]},
                        target_features
                    )
                    song["relevance_score"] = relevance_score
                    unique_mock_songs.append(song)
                    mock_seen_titles.add(song["title"])
            
            # Mock 데이터도 점수 기준으로 정렬
            unique_mock_songs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return unique_mock_songs[:limit]
        
        return recommended_songs

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

        # 2. 감정에 따른 음악 특성 범위 매핑 및 노래 검색 및 랭킹
        recommended_songs = self.get_ranked_songs_by_audio_features(emotion_label, limit=3) 
        
        if recommended_songs:
            logging.info("\n--- Recommended Music List ---")
            for i, song in enumerate(recommended_songs):
                # 로깅 메시지에 URI와 장르 추가
                genres_str = ", ".join(song.get('genres', [])) if song.get('genres') else "N/A"
                logging.info(f"{i+1}. Title: {song['title']}, Artist: {song['artist']}, BPM: {song['bpm']}, Genres: {genres_str}, URI: {song['uri']}, Score: {song.get('relevance_score', 'N/A'):.2f}")
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

    def get_ranked_songs_by_audio_features(self, emotion_label: str, limit: int = 3): # emotion_label 파라미터 추가
        logging.info(f"Mock API call: Simulating search and ranking for songs with audio features for '{emotion_label}' emotion with limit {limit}...")
        time.sleep(1) # 모의 지연
        
        target_features = self.bpm_mapper.get_audio_feature_ranges(emotion_label)

        # Mock 데이터셋을 더 다양하게 구성하고 "(Mock)" 접미사 추가
        # 이제 Mock 데이터도 더 다양한 제목을 가집니다.
        mock_data = [
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
            {"title": "Stairway to Heaven (Mock)", "artist": "Led Zeppelin (Mock)", "bpm": 82, "uri": "#", "genres": ["Rock", "Hard Rock"], "danceability": 25, "acousticness": 60},
            {"title": "Hotel California (Mock)", "artist": "Eagles (Mock)", "bpm": 147, "uri": "#", "genres": ["Rock", "Classic Rock"], "danceability": 50, "acousticness": 40},
            {"title": "Yesterday (Mock)", "artist": "The Beatles (Mock)", "bpm": 94, "uri": "#", "genres": ["Pop", "Rock"], "danceability": 45, "acousticness": 50},
            {"title": "Smells Like Teen Spirit (Mock)", "artist": "Nirvana (Mock)", "bpm": 117, "uri": "#", "genres": ["Grunge", "Rock"], "danceability": 60, "acousticness": 10},
            {"title": "Billie Jean (Mock)", "artist": "Michael Jackson (Mock)", "bpm": 117, "uri": "#", "genres": ["Pop", "Funk"], "danceability": 90, "acousticness": 5},
            {"title": "Like a Rolling Stone (Mock)", "artist": "Bob Dylan (Mock)", "bpm": 95, "uri": "#", "genres": ["Folk Rock"], "danceability": 40, "acousticness": 70},
            {"title": "One (Mock)", "artist": "U2 (Mock)", "bpm": 91, "uri": "#", "genres": ["Rock"], "danceability": 35, "acousticness": 60},
        ]

        # Mock 데이터 필터링, 중복 제거 및 랭킹
        unique_mock_songs = []
        mock_seen_titles = set()
        for song in mock_data:
            is_mock_title_a_genre = False
            if song["title"].lower() in [g.lower() for g in song.get("genres", [])]:
                is_mock_title_a_genre = True

            if song["title"] not in mock_seen_titles and not is_mock_title_a_genre:
                # Mock 데이터에도 relevance_score를 계산하여 추가
                relevance_score = self._calculate_relevance_score(
                    {"tempo": song["bpm"], "danceability": song["danceability"], "acousticness": song["acousticness"]},
                    target_features
                )
                song["relevance_score"] = relevance_score
                unique_mock_songs.append(song)
                mock_seen_titles.add(song["title"])
        
        # Mock 데이터도 점수 기준으로 정렬
        unique_mock_songs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return unique_mock_songs[:limit]


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