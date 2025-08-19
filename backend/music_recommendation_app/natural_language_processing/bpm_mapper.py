# bpm_mapper.py (수정 완료)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BPMMapper:
    """
    '긍정', '부정' 감정 레이블을 음악의 오디오 특성 범위와 매핑하는 클래스입니다.
    """
    def __init__(self):
        """
        긍정/부정 감정과 대표적인 음악 특성 범위 간의 매핑 규칙을 초기화합니다.
        """
        self.emotion_features_map = {
            # 긍정적인 감정은 일반적으로 더 빠르고 춤추기 좋은 음악과 연결됩니다.
            "긍정": {
                "bpm": (110, 150),          # 빠르고 활기찬 템포
                "danceability": (60, 100),    # 춤추기 좋은 정도 (높음)
                "acousticness": (0, 40)       # 전자악기 사용 비중 (어쿠스틱함이 낮음)
            },

            # 부정적인 감정은 일반적으로 더 느리고 차분한 음악과 연결됩니다.
            "부정": {
                "bpm": (60, 90),            # 느리고 차분한 템포
                "danceability": (0, 50),      # 춤추기 좋은 정도 (낮음)
                "acousticness": (60, 100)     # 어쿠스틱 악기 사용 비중 (어쿠스틱함이 높음)
            }
        }
        
        # 매핑되지 않은 감정(예: 오류 발생 시)에 대한 기본값 설정
        self.default_features_range = self.emotion_features_map["긍정"]
        logging.info("BPMMapper가 '긍정'/'부정' 매핑으로 초기화되었습니다.")

    def get_audio_feature_ranges(self, emotion_label: str) -> dict:
        """
        주어진 감정 레이블('긍정' 또는 '부정')에 해당하는 오디오 특성 범위를 반환합니다.
        매핑되지 않은 감정의 경우, 기본값으로 '긍정'의 범위를 반환합니다.

        Args:
            emotion_label (str): SentimentAnalyzer에서 예측한 '긍정' 또는 '부정' 레이블.

        Returns:
            dict: {"bpm": (min, max), "danceability": (min, max), "acousticness": (min, max)} 형태의 딕셔너리.
        """
        features_range = self.emotion_features_map.get(emotion_label)
        
        if features_range:
            logging.info(f"감정 '{emotion_label}'에 대한 음악 특성 범위 반환: {features_range}")
            return features_range
        else:
            logging.warning(f"'{emotion_label}'에 대한 매핑이 없습니다. 기본값(긍정)을 사용합니다.")
            return self.default_features_range

# 이 파일이 직접 실행될 때만 실행되는 테스트 코드
if __name__ == "__main__":
    mapper = BPMMapper()
    
    logging.info("\n--- 오디오 특성 매핑 테스트 ---")
    
    # 긍정 감정 테스트
    logging.info(f"입력: '긍정' -> 출력: {mapper.get_audio_feature_ranges('긍정')}")
    
    # 부정 감정 테스트
    logging.info(f"입력: '부정' -> 출력: {mapper.get_audio_feature_ranges('부정')}")

    # 매핑되지 않은 감정 테스트 (기본값 '긍정' 반환 확인)
    logging.info(f"입력: '알 수 없음' -> 출력: {mapper.get_audio_feature_ranges('알 수 없음')}")