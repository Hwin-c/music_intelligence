import logging
import traceback
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SentimentAnalyzer:
    def __init__(self, model_name="hun3359/klue-bert-base-sentiment"):
        """
        감정 분석 모델을 초기화합니다.
        __init__이 호출될 때 모델과 토크나이저를 즉시 로드합니다. (사전 로딩 방식)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.id2label = None
        
        # ★★★ 핵심 수정: 초기화 시점에 모델을 미리 로드하도록 변경 ★★★
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        모델과 토크나이저를 로드하는 내부 메서드입니다.
        객체 생성 시 한 번만 실행됩니다.
        """
        if self.model is not None:
            logging.info("모델이 이미 로드되어 있습니다.")
            return

        logging.info(f"감정 분석 모델 사전 로드 시작: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logging.info(f"모델을 디바이스 '{self.device}'로 이동 완료.")
            
            self.id2label = self.model.config.id2label
            logging.info(f"SentimentAnalyzer 모델 '{self.model_name}' 사전 로드 성공. {len(self.id2label)}개의 감정 카테고리를 지원합니다.")
            logging.debug(f"모델의 id2label 매핑: {self.id2label}")

        except Exception as e:
            logging.error(f"SentimentAnalyzer 모델 로드 중 치명적인 오류 발생: {e}")
            logging.error(traceback.format_exc())
            raise # 모델 로드 실패는 심각한 문제이므로 예외를 발생시켜 서버 시작을 중단시킴

    def analyze_sentiment(self, text: str):
        """
        주어진 텍스트의 감정을 분석합니다.
        가장 높은 확률을 가진 감정 레이블과 해당 스코어를 반환합니다.
        """
        # 모델은 이미 로드되었으므로, 별도의 로딩 함수 호출이 필요 없습니다.
        logging.info(f"감정 분석 요청 수신: '{text}'")
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_score, predicted_id = torch.max(probabilities, dim=-1)
        
        raw_predicted_id = predicted_id.item()
        
        # 60가지 감정 레이블이 포함된 모델의 id2label 매핑을 사용합니다.
        mapped_label = self.id2label.get(raw_predicted_id, "알 수 없음")
        
        result = {
            "label": mapped_label,
            "score": predicted_score.item(),
            "all_probabilities": probabilities.tolist()[0]
        }
        logging.info(f"텍스트: '{text}', 예측된 감정 결과: {result['label']} (스코어: {result['score']:.4f})")
        return result

# 모델 테스트
if __name__ == "__main__":
    logging.info("--- SentimentAnalyzer 모듈 직접 실행 테스트 시작 ---")
    try:
        # 모델 로딩에 시간이 걸릴 수 있습니다.
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
            # 로그 레벨을 info로 변경하여 기본적으로 보이도록 함
            logging.info(f"텍스트: '{text}'")
            logging.info(f"예측된 레이블: {result['label']} (스코어: {result['score']:.4f})")
        
        logging.info("--- SentimentAnalyzer 모듈 직접 실행 테스트 완료 ---")

    except Exception as e:
        logging.error(f"SentimentAnalyzer 직접 실행 중 오류 발생: {e}")
        logging.error(traceback.format_exc())
        exit(1)