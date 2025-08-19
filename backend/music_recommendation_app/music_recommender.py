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
    logging.debug("Added %s to sys.path for NLP modules.", nlp_dir)

# --- Mock SentimentAnalyzer 및 BPMMapper 클래스들을 먼저 정의합니다. ---
# 이 클래스들은 실제 모듈 임포트가 실패할 경우 사용될 fallback 입니다.
class SentimentAnalyzer: # 기본적으로 Mock 버전으로 시작
    def analyze_sentiment(self, text: str):
        logging.debug("MockSentimentAnalyzer: Analyzing '%s'", text)
        # 긍정 감정 키워드
        if any(keyword in text for keyword in ["신나", "기분 좋", "활기찬", "행복", "즐거워", "기뻐"]):
            return {"label": "신이 난", "score": 0.9} 
        # 슬픔/부정 감정 키워드 (더 많은 변형 포함)
        elif any(keyword in text for keyword in ["슬프", "우울", "비통", "괴로워", "힘들어", "외로워", "눈물", "답답", "씁쓸"]):
            return {"label": "슬픔", "score": 0.8} 
        # 분노 감정 키워드
        elif any(keyword in text for keyword in ["화가 나", "분노", "짜증나", "열받아", "성나"]):
            return {"label": "분노", "score": 0.85}
        # 스트레스 감정 키워드
        elif any(keyword in text for keyword in ["스트레스", "쉬고 싶", "지쳐", "피곤"]):
            return {"label": "스트레스 받는", "score": 0.75}
        # 편안 감정 키워드
        elif any(keyword in text for keyword in ["편안", "느긋", "안도", "평화", "고요"]):
            return {"label": "편안한", "score": 0.75}
        else:
            # 매핑되지 않는 경우 "긍정"으로 기본값 설정 (Mock 테스트 시)
            return {"label": "긍정", "score": 0.5} 

class BPMMapper: # 기본적으로 Mock 버전으로 시작 (bpm_mapper.py의 내용과 동일하게 유지)
    def __init__(self):
        self.emotion_features_map = {
            # --- 긍정적/활기찬 감정 (높은 BPM, 높은 danceability, 낮은 acousticness) ---
            "긍정": {"bpm": (110, 140), "danceability": (70, 100), "acousticness": (0, 30)},
            "기쁨": {"bpm": (120, 140), "danceability": (75, 100), "acousticness": (0, 25)},
            "신이 난": {"bpm": (130, 160), "danceability": (80, 100), "acousticness": (0, 20)},
            "흥분": {"bpm": (140, 170), "danceability": (85, 100), "acousticness": (0, 15)},
            "감사하는": {"bpm": (110, 130), "danceability": (60, 80), "acousticness": (20, 50)},
            "신뢰하는": {"bpm": (100, 120), "danceability": (50, 70), "acousticness": (30, 60)},
            "편안한": {"bpm": (80, 110), "danceability": (40, 60), "acousticness": (50, 80)},
            "만족스러운": {"bpm": (90, 120), "danceability": (50, 75), "acousticness": (40, 70)},
            "느긋": {"bpm": (70, 100), "danceability": (30, 50), "acousticness": (60, 90)},
            "안도": {"bpm": (90, 110), "danceability": (45, 65), "acousticness": (45, 75)},
            "자신하는": {"bpm": (100, 125), "danceability": (60, 85), "acousticness": (10, 40)},

            # --- 부정적/차분한 감정 (낮은 BPM, 낮은 danceability, 높은 acousticness) ---
            "부정": {"bpm": (60, 90), "danceability": (0, 40), "acousticness": (50, 100)},
            "슬픔": {"bpm": (60, 80), "danceability": (0, 30), "acousticness": (60, 100)},
            "우울한": {"bpm": (50, 70), "danceability": (0, 25), "acousticness": (70, 100)},
            "비통한": {"bpm": (40, 60), "danceability": (0, 20), "acousticness": (80, 100)},
            "후회되는": {"bpm": (60, 80), "danceability": (10, 35), "acousticness": (55, 95)},
            "낙담한": {"bpm": (50, 70), "danceability": (0, 25), "acousticness": (70, 100)},
            "마비된": {"bpm": (40, 60), "danceability": (0, 15), "acousticness": (85, 100)},
            "염세적인": {"bpm": (50, 70), "danceability": (0, 30), "acousticness": (65, 95)},
            "눈물이 나는": {"bpm": (60, 85), "danceability": (15, 40), "acousticness": (50, 90)},
            "실망한": {"bpm": (65, 85), "danceability": (20, 45), "acousticness": (45, 85)},
            "환멸을 느끼는": {"bpm": (60, 80), "danceability": (10, 30), "acousticness": (50, 90)},

            # --- 불안/긴장/복잡한 감정 (중간 BPM 또는 특정 분위기) ---
            "불안": {"bpm": (80, 110), "danceability": (30, 70), "acousticness": (20, 60)},
            "두려운": {"bpm": (70, 100), "danceability": (20, 50), "acousticness": (30, 70)},
            "스트레스 받는": {"bpm": (90, 120), "danceability": (40, 70), "acousticness": (10, 50)},
            "취약한": {"bpm": (70, 90), "danceability": (25, 45), "acousticness": (50, 80)},
            "혼란스러운": {"bpm": (80, 110), "danceability": (35, 65), "acousticness": (25, 55)},
            "당혹스러운": {"bpm": (80, 110), "danceability": (35, 65), "acousticness": (25, 55)},
            "회의적인": {"bpm": (70, 95), "danceability": (20, 50), "acousticness": (40, 70)},
            "걱정스러운": {"bpm": (80, 105), "danceability": (30, 60), "acousticness": (30, 65)},
            "조심스러운": {"bpm": (70, 90), "danceability": (25, 45), "acousticness": (50, 80)},
            "초조한": {"bpm": (100, 130), "danceability": (50, 80), "acousticness": (10, 40)},

            # --- 분노/적대적 감정 (강렬하거나 중간 BPM, 낮은 acousticness) ---
            "분노": {"bpm": (120, 150), "danceability": (60, 90), "acousticness": (0, 20)},
            "툴툴대는": {"bpm": (90, 120), "danceability": (40, 70), "acousticness": (10, 50)},
            "좌절한": {"bpm": (90, 120), "danceability": (40, 70), "acousticness": (10, 50)},
            "짜증내는": {"bpm": (100, 130), "danceability": (50, 80), "acousticness": (10, 40)},
            "방어적인": {"bpm": (80, 110), "danceability": (30, 60), "acousticness": (20, 60)},
            "악의적인": {"bpm": (100, 130), "danceability": (50, 80), "acousticness": (10, 40)},
            "안달하는": {"bpm": (110, 140), "danceability": (55, 85), "acousticness": (0, 30)},

            # --- 상처/부정적 관계 감정 ---
            "상처": {"bpm": (60, 90), "danceability": (10, 40), "acousticness": (50, 90)},
            "질투하는": {"bpm": (80, 110), "danceability": (30, 60), "acousticness": (20, 60)},
            "배신당한": {"bpm": (60, 90), "danceability": (10, 40), "acousticness": (50, 90)},
            "고립된": {"bpm": (60, 90), "danceability": (10, 40), "acousticness": (50, 90)},
            "충격 받은": {"bpm": (80, 110), "danceability": (30, 60), "acousticness": (20, 60)},
            "가난한 불우한": {"bpm": (50, 80), "danceability": (0, 30), "acousticness": (60, 100)},
            "희생된": {"bpm": (50, 80), "danceability": (0, 30), "acousticness": (60, 100)},
            "억울한": {"bpm": (70, 100), "danceability": (20, 50), "acousticness": (30, 70)},
            "괴로워하는": {"bpm": (60, 90), "danceability": (10, 40), "acousticness": (50, 90)},

            # --- 당황/부정적 자아 감정 ---
            "고립된(당황한)": {"bpm": (80, 110), "danceability": (35, 65), "acousticness": (25, 55)},
            "남의 시선을 의식하는": {"bpm": (70, 100), "danceability": (25, 55), "acousticness": (40, 70)},
            "외로운": {"bpm": (60, 90), "danceability": (10, 40), "acousticness": (50, 90)},
            "열등감": {"bpm": (70, 100), "danceability": (20, 50), "acousticness": (30, 70)},
            "죄책감의": {"bpm": (60, 90), "danceability": (10, 40), "acousticness": (50, 90)},
            "부끄러운": {"bpm": (70, 95), "danceability": (20, 50), "acousticness": (40, 70)},
            "혐오스러운": {"bpm": (90, 120), "danceability": (40, 70), "acousticness": (10, 50)},
            "한심한": {"bpm": (60, 90), "danceability": (10, 40), "acousticness": (50, 90)},
            "혼란스러운(당황한)": {"bpm": (80, 110), "danceability": (35, 65), "acousticness": (25, 55)},
        }
        
        # '공부' 및 '중립' 감정 제거에 따라 기본 오디오 특성 범위 재설정
        self.default_features_range = {"bpm": (90, 120), "danceability": (40, 80), "acousticness": (20, 70)}
        logging.info("BPM 매퍼 초기화 완료. 기본 오디오 특성 범위: %s", self.default_features_range)

    def get_bpm_range(self, emotion_label: str) -> tuple:
        """
        주어진 감정 레이블에 해당하는 BPM 범위를 반환합니다.
        (이전 버전과의 호환성을 위해 유지)
        """
        features = self.emotion_features_map.get(emotion_label, self.default_features_range)
        return features["bpm"]

    def get_audio_feature_ranges(self, emotion_label: str) -> dict:
        """
        주어진 감정 레이블에 해당하는 오디오 특성(BPM, danceability, acousticness) 범위를 반환합니다.
        매핑되지 않은 감정 레이블의 경우 기본 오디오 특성 범위를 반환합니다.

        Args:
            emotion_label (str): SentimentAnalyzer에서 예측한 감정 레이블.

        Returns:
            dict: {"bpm": (min_bpm, max_bpm), "danceability": (min_dance, max_dance), "acousticness": (min_acoustic, max_acoustic)} 형태의 딕셔너리.
        """
        features_range = self.emotion_features_map.get(emotion_label)
        
        if features_range:
            return features_range
        else:
            logging.warning("'%s' 감정에 대한 오디오 특성 매핑이 정의되지 않았습니다. 기본 범위 %s를 사용합니다.", emotion_label, self.default_features_range)
            return self.default_features_range


# --- 오디오 특성 기반 관련성 점수 계산 함수 (Mock과 실제 모두에서 사용) ---
# 이제 클래스 외부의 독립적인 함수가 되었습니다.
def _calculate_relevance_score(song_data: dict, target_features: dict, weights: dict) -> float:
    """
    노래의 오디오 특성이 목표 범위에 얼마나 잘 부합하는지 점수를 계산합니다.
    각 특성(BPM, Danceability, Acousticness)은 0~1점 사이의 기여도를 가집니다.
    총 점수는 각 특성 점수의 합계입니다.

    Args:
        song_data (dict): 노래의 오디오 특성 데이터 (예: {"tempo": 120, "danceability": 80, "acousticness": 10}).
        target_features (dict): 목표 오디오 특성 범위 (예: {"bpm": (110, 140), ...}).
        weights (dict): 각 특성의 중요도를 나타내는 가중치 (예: {"bpm": 1.0, "danceability": 0.8, ...}).

    Returns:
        float: 계산된 관련성 점수.
    """
    score = 0.0

    # 특성별 점수 계산 도우미 함수
    def calculate_feature_score(song_value, min_target, max_target, is_tempo=False):
        if song_value is None:
            return 0.0
        try:
            # tempo (BPM)만 0-1 스케일링을 고려하고, 나머지는 int로 변환
            # (Getsong API가 danceability, acousticness를 0-100으로 반환한다고 가정)
            if is_tempo and isinstance(song_value, float) and (song_value >= 0 and song_value <= 1):
                song_value = int(song_value * 100)
            else:
                song_value = int(song_value) # 이미 정수이거나 정수로 변환 가능해야 함

            if min_target <= song_value <= max_target:
                center_target = (min_target + max_target) / 2
                range_half = (max_target - min_target) / 2
                if range_half > 0:
                    # 범위 중앙에 가까울수록 높은 점수 (0.5 ~ 1.0)
                    return (1 - abs(song_value - center_target) / range_half) * 0.5 + 0.5
                else: # 범위가 0인 경우 (min=max)
                    return 1.0 if song_value == min_target else 0.0
            return 0.0 # 범위 밖에 있는 경우
        except ValueError:
            return 0.0

    score += calculate_feature_score(song_data.get("tempo"), *target_features["bpm"], is_tempo=True) * weights.get("bpm", 1.0)
    score += calculate_feature_score(song_data.get("danceability"), *target_features["danceability"]) * weights.get("danceability", 1.0)
    score += calculate_feature_score(song_data.get("acousticness"), *target_features["acousticness"]) * weights.get("acousticness", 1.0)
    
    return score

# --- 광범위한 감정 카테고리 매핑 (SentimentAnalyzer 출력 -> 가중치 적용 카테고리) ---
# SentimentAnalyzer가 반환하는 세분화된 감정 레이블을
# 가중치를 정의한 광범위한 감정 카테고리로 매핑합니다.
broad_emotion_category_map = {
    # 긍정
    "긍정": "긍정", "기쁨": "긍정", "신이 난": "긍정", "흥분": "긍정",
    "감사하는": "긍정", "신뢰하는": "긍정", "만족스러운": "긍정", "자신하는": "긍정",

    # 부정
    "부정": "부정", "슬픔": "부정", "우울한": "부정", "비통한": "부정",
    "후회되는": "부정", "낙담한": "부정", "마비된": "부정", "염세적인": "부정",
    "눈물이 나는": "부정", "실망한": "부정", "환멸을 느끼는": "부정",
    "취약한": "부정", "상처": "부정", "질투하는": "부정", "배신당한": "부정",
    "고립된": "부정", "충격 받은": "부정", "가난한 불우한": "부정", "희생된": "부정",
    "억울한": "부정", "괴로워하는": "부정", "외로운": "부정", "열등감": "부정",
    "죄책감의": "부정", "부끄러운": "부정", "한심한": "부정",
    # 이전에 '중립'으로 매핑되었던 감정들을 '부정' 또는 '평온'으로 재분류
    "혼란스러운": "부정", "당혹스러운": "부정", "회의적인": "부정", "조심스러운": "부정",
    "걱정스러운": "부정", 
    "초조한": "부정", 
    "불안": "부정", 
    "고립된(당황한)": "부정", "남의 시선을 의식하는": "부정", "혼란스러운(당황한)": "부정",


    # 평온 (사용자 요청에 따라 '편안한', '느긋', '안도'를 이 카테고리로 매핑)
    "편안한": "평온", "느긋": "평온", "안도": "평온",

    # 분노 (사용자 요청에 따라 '화가 나'를 '분노'로 통일)
    "분노": "분노", "툴툴대는": "분노", "좌절한": "분노", "짜증내는": "분노",
    "방어적인": "분노", "악의적인": "분노", "안달하는": "분노", "혐오스러운": "분노", # 혐오스러운도 분노에 가깝다고 판단

    # 스트레스 (사용자 요청에 따라 추가)
    "스트레스 받는": "스트레스",
}

# --- 감정 카테고리별 오디오 특성 가중치 ---
# 사용자의 요청에 따라 '화가 나' -> '분노', 'neutral' -> '중립'으로 이름 변경
# '공부'와 '중립' 감정은 삭제
emotion_weights = {
    "긍정": { # 행복/신남
        "bpm": 1.0,
        "danceability": 1.0,
        "acousticness": 0.2
    },
    "부정": { # 슬픔/우울
        "bpm": 0.8,
        "danceability": 0.3,
        "acousticness": 1.0
    },
    "평온": { # 평온/휴식
        "bpm": 0.7,
        "danceability": 0.4,
        "acousticness": 0.9
    },
    "분노": { # 분노/격렬
        "bpm": 1.0,
        "danceability": 0.7,
        "acousticness": 0.1
    },
    "스트레스": { # 스트레스 해소 (차분함 위주)
        "bpm": 0.7,
        "danceability": 0.4,
        "acousticness": 0.9
    },
}


class MockMusicRecommender(object):
    """
    API 키가 없거나 개발 시에 사용할 모의(Mock) 추천기입니다.
    실제 API 호출 없이 미리 정의된 데이터를 반환합니다.
    """
    def __init__(self, getsongbpm_api_key: str = None):
        self.sentiment_analyzer = SentimentAnalyzer() # Mock 또는 실제
        self.bpm_mapper = BPMMapper() # Mock 또는 실제
        self.getsongbpm_api_key = getsongbpm_api_key # 사용되지 않지만 일관성을 위해 유지
        logging.info("--- Mock Music Recommender initialized (using mock data only) ---")

    def get_ranked_songs_by_audio_features(self, emotion_label: str, limit: int = 3):
        """
        사용자의 감정에 따라 오디오 특성 기반으로 노래를 검색하고 랭킹을 매겨 반환합니다.
        Mock 데이터만 사용합니다.
        """
        logging.info("Mock API call: Simulating search and ranking for songs with audio features for '%s' emotion with limit %d...", emotion_label, limit)
        time.sleep(1)
        
        target_audio_features = self.bpm_mapper.get_audio_feature_ranges(emotion_label)
        
        # SentimentAnalyzer의 출력 레이블을 광범위한 카테고리로 매핑하여 가중치 선택
        # broad_emotion_category_map에 없는 경우, 기본 가중치 (예: 긍정의 가중치)를 사용하거나 오류 처리
        broad_category = broad_emotion_category_map.get(emotion_label, "긍정") # 기본값을 "긍정"으로 설정
        current_weights = emotion_weights.get(broad_category, emotion_weights["긍정"]) # 기본값을 "긍정"으로 설정

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
                    {"tempo": song["bpm"], "danceability": song["danceability"], "acousticness": song["acousticness"]},
                    target_audio_features,
                    current_weights # 가중치 전달
                )
                song["relevance_score"] = relevance_score
                unique_mock_songs.append(song)
                mock_seen_titles.add(song["title"])
            
        unique_mock_songs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        recommended_songs = unique_mock_songs[:3] # Mock에서는 항상 3개만 반환

        return recommended_songs # MockMusicRecommender의 recommend_music은 이 값을 바로 반환

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
        # Getsong API의 기본 URL만 정의 (텍스트 추천 엔드포인트는 search/로 통일)
        self.getsongbpm_base_url = "https://api.getsong.co/" 
        logging.info("Music recommendation system initialized with API Base URL: %s", self.getsongbpm_base_url)

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
            logging.debug("Calling getsong.co API: %s with params %s", url, params)
            response = requests.get(url, params=params)
            response.raise_for_status() 
            
            response_json = response.json()
            logging.debug("getsong.co API response JSON: %s", json.dumps(response_json, indent=2, ensure_ascii=False))
            
            return response_json
        except requests.exceptions.HTTPError as http_err:
            logging.error("HTTP error occurred: %s - Response Status: %s - Response Text: %s", http_err, response.status_code if response else 'N/A', response.text if response else 'N/A')
        except requests.exceptions.ConnectionError as conn_err:
            logging.error("Network connection error: %s", conn_err)
        except requests.exceptions.Timeout as timeout_err:
            logging.error("Request timeout error: %s", timeout_err)
        except requests.exceptions.RequestException as req_err:
            logging.error("Request error: %s", req_err)
        finally:
            time.sleep(delay_seconds) # API 호출 후 지연 추가
        return None

    def get_ranked_songs_by_audio_features(self, emotion_label: str, limit: int = 3):
        """
        사용자의 감정에 따라 오디오 특성 기반으로 노래를 검색하고 랭킹을 매겨 반환합니다.
        '/search' 엔드포인트를 활용하여 더 다양한 제목을 얻도록 시도합니다.
        """
        logging.info("Attempting to search and rank songs based on audio features for '%s' emotion via getsong.co API...", emotion_label)
        
        target_features = self.bpm_mapper.get_audio_feature_ranges(emotion_label)
        
        # SentimentAnalyzer의 출력 레이블을 광범위한 카테고리로 매핑하여 가중치 선택
        # broad_emotion_category_map에 없는 경우, 기본 가중치 (예: 긍정의 가중치)를 사용하거나 오류 처리
        broad_category = broad_emotion_category_map.get(emotion_label, "긍정") # 기본값을 "긍정"으로 설정
        current_weights = emotion_weights.get(broad_category, emotion_weights["긍정"]) # 기본값을 "긍정"으로 설정

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
            "부정": ["sad", "melancholy", "downbeat", "gloomy", "depressed", "lonely", "heartbreak", "somber", "blue", "sorrow", "grief"], # '비통한' 관련 키워드 추가
            "분노": ["angry", "rage", "intense", "aggressive", "metal", "punk", "hard rock", "rebellion"], 
            "스트레스": ["relax", "calm", "chill", "soothing", "meditation", "peaceful", "unwind"],
            "평온": ["relax", "calm", "chill", "peaceful", "smooth", "mellow", "serene"], 
            "불안": ["soothing", "calm", "meditation", "peaceful", "gentle", "comforting"],
            "비통한": ["grieved", "sorrowful", "heartbroken", "mournful", "despairing", "somber"] 
        }
        
        # 일반적인 장르 키워드 (fallback 용도 또는 추가 다양성)
        genre_keywords = ["pop", "dance", "rock", "electronic", "jazz", "hip hop", "ballad", "r&b", "k-pop", "indie", "soul", "funk", "classical"]

        # 감정 기반 키워드를 우선 사용하고, 장르 키워드를 추가합니다.
        # broad_category_map에 없는 경우, 기본값으로 "긍정"의 키워드를 사용
        keywords_for_query = emotion_keywords_for_search.get(broad_category, emotion_keywords_for_search["긍정"])
        combined_queries = set(keywords_for_query)
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

            # API 응답이 유효하고 'search' 키가 존재하며, 그 값이 리스트인지 확인
            if api_response and isinstance(api_response.get("search"), list): 
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
                        logging.debug("Filtering out song with common title: %s", song_title)
                        continue

                    song_bpm = song_data.get("tempo") 
                    song_danceability = song_data.get("danceability")
                    song_acousticness = song_data.get("acousticness")

                    if song_bpm is not None and song_danceability is not None and song_acousticness is not None:
                        try:
                            # Getsong API에서 받은 값을 int로 변환하여 저장 (HTML 템플릿 사용을 위함)
                            processed_bpm = int(song_bpm)
                            processed_danceability = int(song_danceability)
                            processed_acousticness = int(song_acousticness)

                            relevance_score = _calculate_relevance_score(
                                {"tempo": processed_bpm, "danceability": processed_danceability, "acousticness": processed_acousticness},
                                target_features,
                                current_weights # 가중치 전달
                            )

                            if relevance_score > 0:
                                candidate_songs.append({
                                    "title": song_title,
                                    "artist": artist_name,
                                    "bpm": processed_bpm, # 정수형으로 저장
                                    "uri": song_uri, 
                                    "genres": genres,
                                    "danceability": processed_danceability, # 정수형으로 저장
                                    "acousticness": processed_acousticness, # 정수형으로 저장
                                    "relevance_score": relevance_score
                                })
                                seen_titles.add(song_title)
                        except ValueError:
                            logging.warning("Invalid numeric value for audio feature in song %s. Skipping.", song_title)
                    else:
                        logging.debug("Missing or invalid audio features for song %s. Skipping for ranking.", song_title)
            # 'search' 키가 리스트가 아닌 경우 (예: {"error": "no result"})
            elif api_response and isinstance(api_response.get("search"), dict) and api_response["search"].get("error"):
                logging.warning("API response for query '%s' returned an error: %s. Skipping this query.", query, api_response['search']['error'])
            else:
                logging.warning("API response for query '%s' did not contain a valid list of songs under 'search' key. Response: %s. Skipping this query.", query, api_response)
            
            if len(candidate_songs) >= limit * 5:
                break

        candidate_songs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        recommended_songs = candidate_songs[:limit]

        return recommended_songs # 이 메서드는 추천 노래 목록만 반환합니다.

    def recommend_music(self, user_text: str, limit: int = 3): # limit 매개변수 추가
        """
        사용자 텍스트를 기반으로 음악을 추천하는 메인 메서드입니다.
        감정 분석, 오디오 특성 매핑, 노래 검색 및 랭킹을 포함합니다.
        """
        logging.info("Starting recommendation process for user text: '%s'", user_text)
        
        # 1. 감정 분석
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_text)
        user_emotion = sentiment_result["label"]
        logging.debug("User emotion analyzed as: %s", user_emotion)
        
        # 2. 오디오 특성 범위 매핑
        target_audio_features = self.bpm_mapper.get_audio_feature_ranges(user_emotion)
        logging.debug("Target audio features for '%s': %s", user_emotion, target_audio_features)
        
        # 3. 오디오 특성 기반으로 노래 검색 및 랭킹
        # get_ranked_songs_by_audio_features 메서드를 호출하여 추천 목록을 가져옵니다.
        recommended_songs = self.get_ranked_songs_by_audio_features(user_emotion, limit=limit) # limit 전달

        # 최종 추천 결과 반환 (Flask app.py에서 사용할 구조)
        return {
            "user_emotion": user_emotion,
            "target_audio_features": target_audio_features,
            "recommendations": recommended_songs
        }


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
        "정말 화가 나서 아무것도 못 하겠어.",
        "하루 종일 스트레스 받아서 쉬고 싶어.",
        "너무 편안하고 기분이 좋아.",
        "조금 불안하고 걱정이 되네.",
        "비통하다. 마음이 찢어지는 것 같아.",
        "너무 슬퍼.", # 슬픔 테스트 추가
        "짜증나 죽겠어.", # 분노 테스트 추가
        "마음이 너무 괴로워." # 슬픔 테스트 추가
    ]

    for user_text in test_inputs:
        try:
            result = recommender.recommend_music(user_text)
            logging.info("\nUser Emotion: %s", result['user_emotion'])
            logging.info("Target Audio Features: %s", result['target_audio_features'])
            logging.info("Recommended Songs:")
            for song in result['recommendations']:
                logging.info("  - %s by %s (Score: %.2f)", song['title'], song['artist'], song.get('relevance_score', 0))
            logging.info("\n%s\n", "="*70)
        except Exception as e:
            logging.error("테스트 중 오류 발생: %s", e)
            logging.info("\n%s\n", "="*70)