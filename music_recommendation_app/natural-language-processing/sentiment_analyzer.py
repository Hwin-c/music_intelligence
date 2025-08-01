import logging
import traceback
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification # transformers 임포트를 상단으로 이동

# 로깅 설정 (app.py의 로깅 설정을 따르지만, 이 파일에서도 상세 로그를 위해 DEBUG 레벨 유지)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SentimentAnalyzer:
    def __init__(self, model_name="hun3359/klue-bert-base-sentiment"): # 모델 이름 변경
        """
        감정 분석 모델을 초기화합니다.
        모델은 초기화 시 바로 로드되지 않고, 필요할 때 (첫 analyze_sentiment 호출 시) 로드됩니다.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.id2label = None
        logging.debug(f"SentimentAnalyzer 인스턴스 초기화 완료. 모델 '{self.model_name}'은 지연 로드됩니다.")

    def _load_model_and_tokenizer(self):
        """
        모델과 토크나이저를 실제로 로드하는 내부 메서드입니다.
        이 메서드는 analyze_sentiment가 처음 호출될 때 한 번만 실행됩니다.
        """
        if self.model is not None: # 이미 로드되었다면 다시 로드하지 않음
            return

        logging.debug(f"감정 분석 모델 지연 로드 시작: {self.model_name}")
        try:
            logging.debug("토크나이저 로드 시도 중 (지연 로드)...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logging.debug("토크나이저 로드 완료 (지연 로드).")
            
            logging.debug("감정 분석 모델 로드 시도 중 (지연 로드)...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logging.debug("감정 분석 모델 로드 완료 (지연 로드).")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device) # 모델을 지정된 디바이스로 이동
            logging.debug(f"모델을 디바이스 '{self.device}'로 이동 완료 (지연 로드).")
            
            self.id2label = self.model.config.id2label
            logging.info(f"SentimentAnalyzer 모델 '{self.model_name}' 지연 로드 성공. {len(self.id2label)}개의 감정 카테고리 지원.")
            logging.debug(f"모델의 id2label 매핑: {self.id2label}") # 실제 id2label 값 확인용

        except Exception as e:
            logging.error(f"SentimentAnalyzer 모델 지연 로드 중 치명적인 오류 발생: {e}")
            logging.error(traceback.format_exc())
            raise

    def analyze_sentiment(self, text: str):
        """
        주어진 텍스트의 감정을 분석합니다.
        가장 높은 확률을 가진 감정 레이블과 해당 스코어를 반환합니다.
        """
        self._load_model_and_tokenizer() # 모델이 아직 로드되지 않았다면 여기서 로드

        logging.debug(f"감정 분석 요청 수신: '{text}'")
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_score, predicted_id = torch.max(probabilities, dim=-1)
        
        raw_predicted_id = predicted_id.item() # 예측된 레이블의 숫자 ID
        
        # hun3359/klue-bert-base-sentiment 모델의 id2label 매핑을 직접 사용
        # 이 모델은 긍정/부정/중립 외에 '기쁨', '슬픔', '분노', '불안', '상처', '당황' 등 세분화된 감정을 가집니다.
        mapped_label = self.id2label.get(raw_predicted_id, "알 수 없음") # 모델의 실제 레이블 사용
        
        result = {
            "label": mapped_label, # 매핑된 레이블 사용
            "score": predicted_score.item(),
            "all_probabilities": probabilities.tolist()[0]
        }
        logging.debug(f"텍스트: '{text}', 예측된 감정 결과: {result} (원시 ID: {raw_predicted_id})")
        return result

# 모델 테스트 (이 부분은 파일이 직접 실행될 때만 작동하며, 로깅을 사용하도록 변경)
if __name__ == "__main__":
    logging.info("--- SentimentAnalyzer 모듈 직접 실행 테스트 시작 ---")
    try:
        # transformers 임포트가 이미 상단에 있으므로 여기서 다시 할 필요 없음
        pass 
    except ImportError:
        logging.error("transformers 라이브러리가 설치되어 있지 않습니다. 'pip install transformers'를 실행해주세요.")
        exit(1)

    try:
        analyzer = SentimentAnalyzer()

        logging.info("\n--- 감정 분석 테스트 ---")

        test_cases = [
            "신나는 날이야!", 
            "우울한 날이야", 
            "공부하느라 집중해야 해", 
            "정말 화가 나!", 
            "와우, 깜짝 놀랐어!", 
            "아, 정말 짜증나 죽겠네.", 
            "너무 편안하고 기분이 좋아.", 
            "조금 걱정이 되네.", 
            "나는 지금 행복해", 
            "이 상황은 정말 혼란스럽다."
        ]

        for i, text in enumerate(test_cases):
            logging.info(f"\n--- 테스트 케이스 {i+1} ---")
            result = analyzer.analyze_sentiment(text)
            logging.info(f"텍스트: '{text}'")
            logging.info(f"예측된 레이블: {result['label']} (스코어: {result['score']:.4f})")
            logging.debug(f"모든 확률: {result['all_probabilities']}")
        
        logging.info("--- SentimentAnalyzer 모듈 직접 실행 테스트 완료 ---")

    except Exception as e:
        logging.error(f"SentimentAnalyzer 직접 실행 중 오류 발생: {e}")
        logging.error(traceback.format_exc())
        exit(1)
