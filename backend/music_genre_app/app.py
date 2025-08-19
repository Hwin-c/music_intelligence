# app.py (수정 완료)
from flask import Flask, request, jsonify
from flask_cors import CORS # CORS 라이브러리 임포트
import os
import logging
import traceback
import sys
import threading
import time
import json
import tensorflow as tf
import librosa
import pandas as pd
import numpy as np
import pickle

# Flask 앱 인스턴스 생성 (API 서버이므로 static/template 폴더 설정 불필요)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한 설정

# ======================================================================
# CORS 설정 (Vercel 프론트엔드와의 통신을 위해 필수)
# ======================================================================
# 배포된 Vercel 앱의 URL과 로컬 개발 환경에서의 요청을 허용합니다.
# "origins"에 실제 Vercel 배포 URL을 추가하는 것이 더 안전합니다.
# 예: origins=["http://localhost:3000", "https://your-frontend.vercel.app"]
CORS(app, resources={r"/predict": {"origins": "*"}}) # 우선 모든 출처 허용, 나중에 특정 URL로 제한 가능
logging.info("CORS 설정 완료. /predict 엔드포인트에 대해 모든 출처의 요청을 허용합니다.")
# ======================================================================

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Music Genre App API 서버 초기화 시작.")

# 모델 및 도구 파일의 기본 디렉토리
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# 전역 변수
model = None
scaler = None
genre_labels = None
train_cols = None
model_loaded = False

def _load_genre_classification_models():
    """음악 장르 분류 모델과 관련 도구들을 지연 로드하는 내부 함수."""
    global model, scaler, genre_labels, train_cols, model_loaded
    
    if model_loaded:
        logging.info("모델 및 도구가 이미 로드되어 있습니다.")
        return

    logging.info("음악 장르 분류 모델 및 도구 지연 로드 시작.")
    try:
        keras_model_path = os.path.join(MODEL_DIR, 'model.keras')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        genre_labels_path = os.path.join(MODEL_DIR, 'genre_labels.json')
        feature_columns_path = os.path.join(MODEL_DIR, 'feature_columns.json')

        logging.info(f"Keras 모델 로드: {keras_model_path}")
        model = tf.keras.models.load_model(keras_model_path)

        logging.info(f"스케일러 로드: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        logging.info(f"장르 레이블 로드: {genre_labels_path}")
        with open(genre_labels_path, 'r') as f:
            genre_labels = json.load(f)
        
        logging.info(f"피처 컬럼 로드: {feature_columns_path}")
        with open(feature_columns_path, 'r') as f:
            train_cols = json.load(f)

        model_loaded = True
        logging.info("음악 장르 분류 모델 및 도구 로드 성공.")
    except Exception as e:
        logging.error(f"모델 로드 중 치명적 오류 발생: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

# 앱 컨텍스트 내에서 모델 로드 실행
with app.app_context():
    _load_genre_classification_models()

def extract_features_from_audio(file_path):
    """오디오 파일에서 특징을 추출하는 함수 (기존 코드와 동일)"""
    try:
        y, sr = librosa.load(file_path, sr=44100)
        max_samples = int(30 * sr)
        if len(y) > max_samples:
            y = y[:max_samples]
        
        if np.isnan(y).any():
            raise ValueError("오디오 신호에 NaN 값이 포함되어 있습니다.")
        
        features = {}
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_var'] = np.var(chroma_stft)
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spec_centroid)
        features['spectral_centroid_var'] = np.var(spec_centroid)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spec_bandwidth)
        features['spectral_bandwidth_var'] = np.var(spec_bandwidth)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_var'] = np.var(rolloff)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_var'] = np.var(zcr)
        harmony = librosa.effects.harmonic(y)
        features['harmony_mean'] = np.mean(harmony)
        features['harmony_var'] = np.var(harmony)
        perceptr = librosa.feature.spectral_flatness(y=y)
        features['perceptr_mean'] = np.mean(perceptr)
        features['perceptr_var'] = np.var(perceptr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_var'] = np.var(mfcc[i])

        return pd.DataFrame([features])
    except Exception as e:
        logging.error(f"오디오 특징 추출 중 오류: {e}")
        logging.error(traceback.format_exc())
        raise

@app.route('/healthz')
def health_check():
    """헬스 체크 엔드포인트. 서비스 상태를 확인합니다."""
    if model_loaded:
        return "OK", 200
    else:
        return "Model not loaded", 503

@app.route('/predict', methods=['POST'])
def predict_genre_endpoint():
    """오디오 파일을 받아 장르를 예측하고 JSON으로 결과를 반환하는 API 엔드포인트."""
    if not model_loaded:
        logging.error("예측 요청 실패: 모델이 로드되지 않았습니다.")
        return jsonify({'error': 'Model is not loaded yet. Please try again later.'}), 503

    if 'audio' not in request.files:
        logging.warning("API 경고: 'audio' 필드 없이 파일 업로드 요청이 들어왔습니다.")
        return jsonify({'error': 'No audio file part in the request'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        logging.warning("API 경고: 파일 이름이 없는 빈 파일 파트가 제출되었습니다.")
        return jsonify({'error': 'No selected file'}), 400

    filename = audio_file.filename.lower()
    if not filename.endswith('.wav'):
        logging.warning(f"지원하지 않는 파일 형식 시도: {filename}")
        return jsonify({'error': 'Unsupported file format. Please upload a WAV file.'}), 400

    temp_wav_path = os.path.join(MODEL_DIR, f'temp_{os.getpid()}_{threading.get_ident()}.wav')
    
    try:
        audio_file.save(temp_wav_path)
        
        features_df = extract_features_from_audio(temp_wav_path)
        
        processed_features = pd.DataFrame(columns=train_cols)
        for col in train_cols:
            processed_features[col] = features_df.get(col, 0.0)
        
        processed_features = processed_features.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        features_scaled = scaler.transform(processed_features)
        
        preds = model.predict(features_scaled)
        
        label_idx = np.argmax(preds)
        label = genre_labels[label_idx]
        probability = float(preds[0][label_idx]) * 100
        
        logging.info(f"예측 성공: {label} ({probability:.2f}%)")
        return jsonify({'label': label, 'probability': f"{probability:.2f}"})

    except Exception as e:
        logging.error(f'API 오류: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

# 로컬 개발 환경에서 직접 실행할 때 사용됩니다.
# Gunicorn으로 실행될 때는 이 부분이 사용되지 않습니다.
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # debug=False로 설정하여 프로덕션 환경과 유사하게 테스트
    app.run(host='0.0.0.0', port=port, debug=False)