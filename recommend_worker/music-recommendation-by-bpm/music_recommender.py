import requests
import random
import os
import sys # sys.path 조작을 위해 임포트

# 부모 디렉토리를 Python 경로에 추가하여 natural_language_processing 모듈을 임포트할 수 있도록 합니다.
# 이 스크립트가 music-recommendation-by-bpm/ 에 위치하므로,
# natural_language_processing 모듈은 상대 경로로 접근 가능합니다.
# 하지만 직접 실행 환경에서는 sys.path 추가가 더 안전합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
# music-recommendation-by-bpm/natural-language-processing/ 이 경로를 sys.path에 추가해야 합니다.
# current_dir이 music-recommendation-by-bpm/ 이므로, nlp_dir는 current_dir/natural-language-processing 입니다.
nlp_dir = os.path.join(current_dir, 'natural-language-processing')

if nlp_dir not in sys.path:
    sys.path.append(nlp_dir)

try:
    # 수정: 이제 natural_language_processing 디렉토리가 sys.path에 있으므로,
    # 해당 디렉토리 내부의 모듈을 직접 임포트합니다.
    from sentiment_analyzer import SentimentAnalyzer
    from bpm_mapper import BPMMapper
except ImportError as e:
    print(f"필수 모듈 임포트 실패: {e}")
    print("natural-language-processing 디렉토리가 올바른 경로에 있는지 확인하고,")
    print("혹은 'python -m pip install -e .' 와 같은 개발 모드 설치가 필요한지 확인하십시오.")
    sys.exit(1)


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
        # getsongbpm API의 실제 검색 엔드포인트와 인증 방식에 따라 수정 필요
        self.getsongbpm_base_url = "https://api.getsongbpm.com"
        print("음악 추천 시스템 초기화 완료.")

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
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status() # 200 이외의 응답 시 예외 발생
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP 오류 발생: {http_err} - 응답: {response.text}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"네트워크 연결 오류: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"요청 시간 초과 오류: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"요청 오류: {req_err}")
        return None

    def get_songs_by_bpm_range(self, min_bpm: int, max_bpm: int, limit: int = 5):
        """
        getsongbpm API를 사용하여 특정 BPM 범위 내의 노래를 검색합니다.
        
        **주의:** getsongbpm API는 직접적인 'BPM 범위 검색' 기능을 명시적으로 제공하지 않을 수 있습니다.
        이 함수는 해당 API가 트랙 검색 결과를 반환하고, 그 결과에 BPM 정보가 포함되어 있음을
        가정하여 클라이언트 측에서 BPM으로 필터링하는 방식으로 구현됩니다.
        실제 API 문서를 확인하여 가장 효율적인 검색 방식을 적용해야 합니다.
        (예: 장르 또는 인기곡 검색 후 BPM 필터링)
        """
        print(f"getsongbpm API에서 BPM {min_bpm}~{max_bpm} 범위의 노래 검색 시도 중...")
        
        print("getsongbpm API의 직접적인 BPM 범위 검색은 제한적일 수 있습니다.")
        print("임시로 BPM 필터링이 가능한 샘플 데이터를 시뮬레이션합니다.")
        
        mock_songs_data = [
            {"title": "Happy Beats", "artist": "Joyful Band", "bpm": 135},
            {"title": "Sad Melody", "artist": "Lonely Singer", "bpm": 65},
            {"title": "Focus Zone", "artist": "Study Tunes", "bpm": 105},
            {"title": "Energetic Rush", "artist": "Power Duo", "bpm": 150},
            {"title": "Chill Out", "artist": "Relax Vibes", "bpm": 85},
            {"title": "Midnight Blues", "artist": "Jazz Master", "bpm": 70},
            {"title": "Upbeat Pop", "artist": "Pop Star", "bpm": 125},
            {"title": "Calm Study", "artist": "Concentration Crew", "bpm": 95},
            {"title": "Intense Rock", "artist": "Hardcore Band", "bpm": 145},
            {"title": "Mellow Jazz", "artist": "Smooth Sounds", "bpm": 80},
        ]
        
        filtered_songs = [
            song for song in mock_songs_data 
            if min_bpm <= song["bpm"] <= max_bpm
        ]
        
        if not filtered_songs:
            print("경고: 시뮬레이션 데이터에서 해당 BPM 범위의 곡을 찾지 못했습니다. 임의의 곡을 반환합니다.")
            return random.sample(mock_songs_data, min(limit, len(mock_songs_data)))
            
        return random.sample(filtered_songs, min(limit, len(filtered_songs)))


    def recommend_music(self, user_text: str):
        """
        사용자 텍스트를 기반으로 음악을 추천합니다.

        Args:
            user_text (str): 사용자의 감정 표현 텍스트.

        Returns:
            list: 추천된 음악 리스트 (제목, 아티스트, BPM 포함).
        """
        print(f"\n--- '{user_text}'에 대한 음악 추천 시작 ---")
        
        # 1. 감정 분석
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_text)
        emotion_label = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]
        
        print(f"감정 분석 결과: '{emotion_label}' (스코어: {sentiment_score:.4f})")

        # 2. 감정에 따른 BPM 범위 매핑
        min_bpm, max_bpm = self.bpm_mapper.get_bpm_range(emotion_label)
        print(f"'{emotion_label}' 감정에 추천되는 BPM 범위: {min_bpm}-{max_bpm}")

        # 3. getsongbpm API를 통해 노래 검색 (또는 시뮬레이션 데이터 사용)
        recommended_songs = self.get_songs_by_bpm_range(min_bpm, max_bpm, limit=5)
        
        if recommended_songs:
            print("\n--- 추천 음악 리스트 ---")
            for i, song in enumerate(recommended_songs):
                print(f"{i+1}. 제목: {song['title']}, 아티스트: {song['artist']}, BPM: {song['bpm']}")
        else:
            print("\n죄송합니다. 현재 BPM 범위에 맞는 음악을 찾을 수 없습니다.")
            print("다른 감정을 입력해 보시거나 나중에 다시 시도해주세요.")
            
        return recommended_songs

# 이 파일이 직접 실행될 때만 실행되는 테스트 코드
if __name__ == "__main__":
    # getsongbpm API 키는 환경 변수로 관리하는 것이 보안상 안전합니다.
    # 환경 변수 GETSONGBPM_API_KEY가 설정되어 있지 않으면 "YOUR_GETSONGBPM_API_KEY_HERE"를 사용합니다.
    getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY", "YOUR_GETSONGBPM_API_KEY_HERE")

    if getsongbpm_api_key == "YOUR_GETSONGBPM_API_KEY_HERE":
        print("\n[경고]: getsongbpm API 키가 설정되지 않았습니다.")
        print("실제 API 호출 대신 모의(Mock) 데이터를 사용하여 테스트를 진행합니다.")
        
        class MockMusicRecommender(MusicRecommender):
            def __init__(self, getsongbpm_api_key: str):
                # 부모 클래스의 __init__ 호출 시 getsongbpm_api_key를 전달해야 합니다.
                super().__init__(getsongbpm_api_key) 
                print("--- Mock Music Recommender 초기화 완료 ---")

            def get_songs_by_bpm_range(self, min_bpm: int, max_bpm: int, limit: int = 5):
                print(f"Mock API 호출: BPM {min_bpm}~{max_bpm} 범위 노래 시뮬레이션 검색 중...")
                
                mock_songs_data = [
                    {"title": "기분 좋은 아침", "artist": "김미소", "bpm": 130},
                    {"title": "고요한 숲길", "artist": "이평화", "bpm": 75},
                    {"title": "집중의 순간", "artist": "박몰입", "bpm": 100},
                    {"title": "파워 업!", "artist": "최에너지", "bpm": 155},
                    {"title": "차분한 저녁", "artist": "정고요", "bpm": 60},
                    {"title": "활기찬 하루", "artist": "강다이나믹", "bpm": 120},
                    {"title": "생각의 흐름", "artist": "윤명상", "bpm": 90},
                    {"title": "분노의 질주", "artist": "서파워", "bpm": 140},
                    {"title": "슬픈 빗소리", "artist": "오감성", "bpm": 70},
                    {"title": "새로운 도전", "artist": "조열정", "bpm": 115},
                ]
                
                filtered_songs = [
                    song for song in mock_songs_data 
                    if min_bpm <= song["bpm"] <= max_bpm
                ]
                
                if not filtered_songs:
                    print(f"  Mock 데이터에서 BPM {min_bpm}~{max_bpm} 범위의 곡을 찾지 못했습니다. 임의의 Mock 곡을 반환합니다.")
                    return random.sample(mock_songs_data, min(limit, len(mock_songs_data)))
                
                return random.sample(filtered_songs, min(limit, len(filtered_songs)))

        recommender = MockMusicRecommender(getsongbpm_api_key) # Mock 클래스 사용
    else:
        recommender = MusicRecommender(getsongbpm_api_key) # 실제 API 키로 MusicRecommender 사용

    print("\n==== 음악 추천 시스템 데모 시작 ====")

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
        print("\n" + "="*70 + "\n")