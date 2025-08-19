# sentiment_analyzer.py (수정 완료)

import logging
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SentimentAnalyzer:
    def __init__(self, model_name="sangrimlee/bert-base-multilingual-cased-nsmc"):
        """
        감정 분석 모델을 초기화합니다.
        모델은 필요할 때 (첫 analyze_sentiment 호출 시) 지연 로드됩니다.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        # 이 모델은 '부정(negative)'을 0, '긍정(positive)'을 1로 출력합니다.
        # id2label을 직접 정의합니다.
        self.id2label = {0: '부정', 1: '긍정'}
        logging.info(f"SentimentAnalyzer가 '{self.model_name}' 모델을 사용하도록 초기화되었습니다.")

    def _load_model_and_tokenizer(self):
        """
        모델과 토크나이저를 실제로 로드하는 내부 메서드입니다.
        """
        if self.model is not None:
            return

        logging.info(f"감정 분석 모델 지연 로드 시작: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            logging.info(f"모델 로드 성공. 디바이스 '{self.device}'를 사용합니다.")

        except Exception as e:
            logging.error(f"모델 로드 중 치명적인 오류 발생: {e}")
            logging.error(traceback.format_exc())
            raise

    def analyze_sentiment(self, text: str):
        """
        주어진 텍스트의 감정을 '긍정' 또는 '부정'으로 분석합니다.
        """
        self._load_model_and_tokenizer()

        logging.info(f"감정 분석 요청: '{text}'")
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            predicted_score, predicted_id_tensor = torch.max(probabilities, dim=-1)
            predicted_id = predicted_id_tensor.item()
            
            # 숫자 ID(0 또는 1)를 우리가 정의한 '부정' 또는 '긍정' 문자열로 변환
            mapped_label = self.id2label.get(predicted_id, "알 수 없음")
            
            result = {
                "label": mapped_label,
                "score": predicted_score.item()
            }
            logging.info(f"분석 결과: '{mapped_label}' (점수: {result['score']:.4f})")
            return result

        except Exception as e:
            logging.error(f"'{text}' 감정 분석 중 오류 발생: {e}", exc_info=True)
            # 오류 발생 시 기본값 반환
            return {"label": "긍정", "score": 0.5}

# 이 파일이 직접 실행될 때만 실행되는 테스트 코드
if __name__ == "__main__":
    logging.info("--- SentimentAnalyzer 모듈 직접 실행 테스트 시작 ---")
    
    analyzer = SentimentAnalyzer()

    test_cases = [
        "이 영화 정말 최고예요! 배우들 연기가 너무 인상 깊었어요.",
        "정말 실망스러운 결말이었어요. 시간이 아깝네요.",
        "평범한데, 딱히 재미있지도 재미없지도 않음.",
        "우울할 때 들으면 위로가 되는 노래",
        "스트레스 확 풀리는 신나는 음악 추천해줘"
    ]

    for text in test_cases:
        analyzer.analyze_sentiment(text)
        print("-" * 20)

    logging.info("--- SentimentAnalyzer 모듈 직접 실행 테스트 완료 ---")