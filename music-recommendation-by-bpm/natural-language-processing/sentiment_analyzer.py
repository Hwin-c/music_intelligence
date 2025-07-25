import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch # torch는 여전히 텐서 연산을 위해 필요합니다.

class SentimentAnalyzer:
    def __init__(self, model_name="hun3359/klue-bert-base-sentiment"):
        """
        감정 분석 모델을 초기화합니다.
        모델은 초기화 시 한 번만 로드됩니다.
        """
        print(f"감정 분석 모델 로드 중: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # 모델의 config에서 id2label 매핑 정보 로드
        self.id2label = self.model.config.id2label
        
        print("모델 로드 성공.")
        print(f"모델은 총 {len(self.id2label)}개의 감정 카테고리를 지원합니다.")

    def analyze_sentiment(self, text: str):
        """
        주어진 텍스트의 감정을 분석합니다.
        가장 높은 확률을 가진 감정 레이블과 해당 스코어를 반환합니다.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad(): # 추론 시에는 기울기 계산을 비활성화하여 메모리 및 성능 최적화
            outputs = self.model(**inputs) # inputs는 기본적으로 CPU 텐서로 처리됩니다.
        
        # Logits에서 softmax를 적용하여 확률 분포 얻기
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 가장 높은 확률을 가진 레이블의 인덱스와 스코어 찾기
        predicted_score, predicted_id = torch.max(probabilities, dim=-1)
        
        predicted_label = self.id2label.get(predicted_id.item(), "unknown")
        
        # 결과를 딕셔너리 형태로 반환
        return {
            "label": predicted_label,
            "score": predicted_score.item(),
            "all_probabilities": probabilities.tolist()[0]
        }

# 모델 테스트 (이 부분은 파일이 직접 실행될 때만 작동)
if __name__ == "__main__":
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        print("transformers 및 torch 라이브러리가 설치되어 있지 않습니다. 'pip install transformers torch'를 실행해주세요.")
        exit(1)

    analyzer = SentimentAnalyzer()

    print("\n--- 감정 분석 테스트 ---")

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
        print(f"\n--- 테스트 케이스 {i+1} ---")
        result = analyzer.analyze_sentiment(text)
        print(f"텍스트: '{text}'")
        print(f"예측된 레이블: {result['label']} (스코어: {result['score']:.4f})")