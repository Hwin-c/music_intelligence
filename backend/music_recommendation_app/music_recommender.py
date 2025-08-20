# music_recommender.py (최종 수정 완료)

import requests
import random
import os
import logging
import time
import json

# ======================================================================
# ★★★★★ 수정: 상대 경로 임포트('.')를 제거하여 절대 경로 임포트로 변경 ★★★★★
# ======================================================================
from natural_language_processing.sentiment_analyzer import SentimentAnalyzer
from natural_language_processing.bpm_mapper import BPMMapper
# ======================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# _calculate_relevance_score 함수 (변경 없음)
def _calculate_relevance_score(song_data: dict, target_features: dict, weights: dict, sentiment_score: float) -> float:
    audio_feature_score = 0.0
    def calculate_feature_score(song_value, min_target, max_target):
        if song_value is None: return 0.0
        try:
            song_value = int(song_value)
            if min_target <= song_value <= max_target:
                center_target = (min_target + max_target) / 2
                range_half = (max_target - min_target) / 2
                if range_half > 0:
                    return (1 - abs(song_value - center_target) / range_half) * 0.5 + 0.5
                return 1.0 if song_value == min_target else 0.0
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    audio_feature_score += calculate_feature_score(song_data.get("tempo"), *target_features["bpm"]) * weights.get("bpm", 1.0)
    audio_feature_score += calculate_feature_score(song_data.get("danceability"), *target_features["danceability"]) * weights.get("danceability", 1.0)
    audio_feature_score += calculate_feature_score(song_data.get("acousticness"), *target_features["acousticness"]) * weights.get("acousticness", 1.0)
    
    total_weight = sum(weights.values())
    if total_weight == 0: return 0.0

    normalized_audio_score = audio_feature_score / total_weight
    final_score = normalized_audio_score * sentiment_score
    
    return final_score

# emotion_weights 딕셔너리 (변경 없음)
emotion_weights = {
    "긍정": {"bpm": 1.0, "danceability": 1.0, "acousticness": 0.2},
    "부정": {"bpm": 0.8, "danceability": 0.3, "acousticness": 1.0},
}


# MusicRecommender 클래스 (내부 로직 변경 없음)
class MusicRecommender:
    def __init__(self, getsongbpm_api_key: str):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.bpm_mapper = BPMMapper()
        self.getsongbpm_api_key = getsongbpm_api_key
        self.getsongbpm_base_url = "https://api.getsong.co/"
        
        ### ★★★ 핵심 수정 사항: 모델 사전 로딩 (Pre-loading) ★★★ ###
        # 서버 시작 시 무거운 NLP 모델을 미리 메모리에 로드하여,
        # 첫 번째 사용자 요청 시 발생하는 긴 지연(타임아웃)을 방지합니다.
        try:
            start_time = time.time()
            logging.info("모델 예열 시작: NLP 모델을 메모리에 사전 로딩합니다...")
            # 간단한 텍스트로 감정 분석을 1회 실행하여 모델 로딩을 강제합니다.
            self.sentiment_analyzer.analyze_sentiment("모델 초기화")
            end_time = time.time()
            logging.info(f"모델 예열 완료. (소요 시간: {end_time - start_time:.2f}초)")
        except Exception as e:
            # 예열 중 오류가 발생하면 심각한 문제이므로 CRITICAL 레벨로 로깅합니다.
            logging.critical(f"모델 예열 실패: {e}", exc_info=True)
            # 서버가 시작되지 않도록 예외를 다시 발생시킬 수도 있습니다.
            # raise e 
        ### ★★★ 수정 완료 ★★★ ###
            
        logging.info("음악 추천 시스템이 성공적으로 초기화되었습니다.")

    def _call_getsongbpm_api(self, endpoint: str, params: dict = None):
        if not self.getsongbpm_api_key:
            logging.warning("Getsong API 키가 없어 API를 호출할 수 없습니다.")
            return None
        
        params = params or {}
        params["api_key"] = self.getsongbpm_api_key
        url = f"{self.getsongbpm_base_url}{endpoint.lstrip('/')}"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            # API 호출 간격을 약간 늘려 안정성을 확보합니다.
            time.sleep(1.5)
            return response.json()
        except requests.exceptions.RequestException as err:
            logging.error(f"Getsong API 호출 실패: {err}")
            return None

    def get_ranked_songs_by_audio_features(self, emotion_label: str, sentiment_score: float, limit: int = 3):
        logging.info(f"'{emotion_label}' 감정(점수: {sentiment_score:.2f})에 맞는 노래를 검색합니다...")
        target_features = self.bpm_mapper.get_audio_feature_ranges(emotion_label)
        
        current_weights = emotion_weights.get(emotion_label, emotion_weights["긍정"])
        emotion_keywords_for_search = {
            "긍정": ["happy", "upbeat", "energetic", "party", "dancing"],
            "부정": ["sad", "melancholy", "downbeat", "gloomy", "heartbreak", "chill"],
        }
        keywords_for_query = emotion_keywords_for_search.get(emotion_label, emotion_keywords_for_search["긍정"])
        
        candidate_songs = []
        seen_titles = set()
        
        random.shuffle(keywords_for_query)

        for query in keywords_for_query[:3]:
            params = {"type": "song", "lookup": query, "limit": 50}
            api_response = self._call_getsongbpm_api("search/", params)
            
            if api_response and isinstance(api_response.get("search"), list):
                for song_data in api_response["search"]:
                    title = song_data.get("title")
                    if not title or title.lower() in seen_titles: continue
                    
                    artist_name = song_data.get("artist", {}).get("name", "Unknown Artist")
                    
                    if not all(k in song_data for k in ["tempo", "danceability", "acousticness"]):
                        continue

                    relevance_score = _calculate_relevance_score(song_data, target_features, current_weights, sentiment_score)

                    if relevance_score > 0.1:
                        candidate_songs.append({
                            "title": title, "artist": artist_name,
                            "bpm": int(song_data["tempo"]),
                            "uri": song_data.get("uri", "#"),
                            "genres": song_data.get("artist", {}).get("genres", []),
                            "danceability": int(song_data["danceability"]),
                            "acousticness": int(song_data["acousticness"]),
                            "relevance_score": relevance_score
                        })
                        seen_titles.add(title.lower())
            
            if len(candidate_songs) >= limit * 5: break
        
        candidate_songs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return candidate_songs[:limit]

    def recommend_music(self, user_text: str, limit: int = 3):
        logging.info(f"추천 프로세스 시작: '{user_text}'")
        # 이제 이 메소드는 모델을 로드하지 않으므로 매우 빠르게 실행됩니다.
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_text)
        user_emotion = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]

        target_audio_features = self.bpm_mapper.get_audio_feature_ranges(user_emotion)
        
        recommended_songs = self.get_ranked_songs_by_audio_features(user_emotion, sentiment_score, limit=limit)
        
        return {
            "user_emotion": user_emotion,
            "target_audio_features": target_audio_features,
            "recommendations": recommended_songs
        }


# 테스트 코드 (변경 없음)
if __name__ == "__main__":
    import os
    getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY")

    if not getsongbpm_api_key:
        logging.warning("\n[테스트 경고]: GETSONGBPM_API_KEY 환경 변수가 없어 실제 API 테스트를 건너뜁니다.")
    else:
        recommender = MusicRecommender(getsongbpm_api_key)
        logging.info("\n==== 음악 추천 시스템 데모 시작 ====")
        test_inputs = [
            "오늘은 정말 기분이 좋고 신나는 하루였어!",
            "나쁘진 않은데 그냥 그래.",
            "너무 우울해서 아무것도 하기 싫다.",
            "이 영화는 내 인생 최악의 영화다.",
        ]
        for user_text in test_inputs:
            result = recommender.recommend_music(user_text)
            logging.info(f"\n>> 사용자 입력: '{user_text}'")
            logging.info(f"   분석된 감정: {result['user_emotion']}")
            logging.info("   추천된 노래:")
            for song in result['recommendations']:
                logging.info(f"     - {song['title']} by {song['artist']} (최종 점수: {song.get('relevance_score', 0):.4f})")