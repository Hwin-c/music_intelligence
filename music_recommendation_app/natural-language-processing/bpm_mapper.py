import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class BPMMapper:
    """
    감정 레이블을 음악의 BPM 범위 및 기타 오디오 특성(danceability, acousticness)과 매핑하는 클래스입니다.
    사용자의 감정에 따라 적절한 음악 특성 범위를 제공합니다.
    """
    def __init__(self):
        """
        감정 레이블과 BPM/오디오 특성 범위 간의 매핑 규칙을 초기화합니다.
        이 매핑은 모델의 60가지 세분화된 감정 레이블과
        일반적인 음악 분위기 및 BPM, danceability, acousticness 상관관계를 기반으로 정의되었습니다.
        여기에 '긍정'과 '부정'과 같은 더 넓은 범주의 감정 매핑을 추가합니다.
        """
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
        
        # 매핑되지 않은 감정에 대한 기본 오디오 특성 범위 설정
        self.default_features_range = {"bpm": (90, 120), "danceability": (40, 80), "acousticness": (20, 70)}
        logging.info(f"BPM 매퍼 초기화 완료. 기본 오디오 특성 범위: {self.default_features_range}")

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
            logging.warning(f"'{emotion_label}' 감정에 대한 오디오 특성 매핑이 정의되지 않았습니다. 기본 범위 {self.default_features_range}를 사용합니다.")
            return self.default_features_range

# 이 파일이 직접 실행될 때만 실행되는 테스트 코드
if __name__ == "__main__":
    mapper = BPMMapper()
    
    logging.info("\n--- 오디오 특성 매핑 테스트 ---")
    
    # 긍정적/활기찬 감정 테스트
    logging.info(f"감정: '긍정' -> 특성: {mapper.get_audio_feature_ranges('긍정')}")
    logging.info(f"감정: '신이 난' -> 특성: {mapper.get_audio_feature_ranges('신이 난')}")
    
    # 부정적/차분한 감정 테스트
    logging.info(f"감정: '슬픔' -> 특성: {mapper.get_audio_feature_ranges('슬픔')}")
    logging.info(f"감정: '우울한' -> 특성: {mapper.get_audio_feature_ranges('우울한')}")

    # 매핑되지 않은 감정 테스트 (기본값 반환 확인)
    logging.info(f"감정: '알 수 없는 감정' -> 특성: {mapper.get_audio_feature_ranges('알 수 없는 감정')}")