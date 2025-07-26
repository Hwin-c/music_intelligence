from flask import Flask, request, jsonify
import os
import librosa
import pandas as pd
import numpy as np
import tensorflow.lite as tflite
import pickle
import json
from pydub import AudioSegment
import logging
import traceback
import sys

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한 설정

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Genre Worker 초기화 시작.")

# --- 음악 장르 분류 기능 관련 모델 및 도구 로드 (앱 시작 시 로드) ---
# 이 워커는 장르 분류만 담당하므로, 모델을 시작 시점에 로드합니다.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'music-genre-classification', 'saved_models')

interpreter = None
input_details = None
output_details = None
scaler = None
genre_labels = None
train_cols = None

def _load_genre_classification_models_on_startup():
    """
    음악 장르 분류 모델과 관련 도구들을 앱 시작 시 로드하는 함수.
    """
    global interpreter, input_details, output_details, scaler, genre_labels, train_cols

    logging.debug("음악 장르 분류 모델 및 도구 로드 시작 (워커 시작 시).")
    try:
        tflite_model_path = os.path.join(MODEL_DIR, 'quantized_model.tflite')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        genre_labels_path = os.path.join(MODEL_DIR, 'genre_labels.json')
        feature_columns_path = os.path.join(MODEL_DIR, 'feature_columns.json')

        logging.debug(f"TFLite 모델 로드 시도 중: {tflite_model_path}")
        interpreter = tflite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logging.debug("TFLite 모델 로드 및 텐서 할당 완료.")

        logging.debug(f"스케일러 로드 시도 중: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logging.debug("스케일러 로드 완료.")

        logging.debug(f"장르 레이블 로드 시도 중: {genre_labels_path}")
        with open(genre_labels_path, 'r') as f:
            genre_labels = json.load(f)
        logging.debug("장르 레이블 로드 완료.")

        logging.debug(f"피처 컬럼 로드 시도 중: {feature_columns_path}")
        with open(feature_columns_path, 'r') as f:
            train_cols = json.load(f)
        logging.debug("피처 컬럼 로드 완료.")

        logging.info("음악 장르 분류 TFLite 모델 및 도구 로드 성공 (워커 시작 시).")
    except Exception as e:
        logging.error(f"음악 장르 분류 모델 로드 중 오류 발생: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)  # 모델 로드 실패 시 워커 종료

# 워커 시작 시 모델 로드
_load_genre_classification_models_on_startup()

def extract_features_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=44100)

    if np.isnan(y).any():
        raise ValueError("오디오 신호가 유효하지 않습니다.")

    logging.info(f"Audio loaded: duration={len(y)/sr:.2f} seconds, sr={sr}")

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony = librosa.effects.harmonic(y)
    perceptr = librosa.feature.spectral_flatness(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = {
        'chroma_stft_mean': np.mean(chroma_stft),
        'chroma_stft_var': np.var(chroma_stft),
        'rms_mean': np.mean(rms),
        'rms_var': np.var(rms),
        'spectral_centroid_mean': np.mean(spec_centroid),
        'spectral_centroid_var': np.var(spec_centroid),
        'spectral_bandwidth_mean': np.mean(spec_bandwidth),
        'spectral_bandwidth_var': np.var(spec_bandwidth),
        'rolloff_mean': np.mean(rolloff),
        'rolloff_var': np.var(rolloff),
        'zero_crossing_rate_mean': np.mean(zcr),
        'zero_crossing_rate_var': np.var(zcr),
        'harmony_mean': np.mean(harmony),
        'harmony_var': np.var(harmony),
        'perceptr_mean': np.mean(perceptr),
        'perceptr_var': np.var(perceptr),
        'tempo': tempo,
    }

    for i in range(20):
        features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
        features[f'mfcc{i+1}_var'] = np.var(mfcc[i])

    return pd.DataFrame([features])

@app.route('/healthz')
def health_check():
    """워커 헬스 체크를 위한 엔드포인트."""
    logging.debug("Genre Worker Health check 요청 수신.")
    return "OK", 200

@app.route('/predict_genre', methods=['POST'])
def predict_genre_endpoint():
    """
    오디오 파일을 입력받아 장르를 예측하는 엔드포인트입니다.
    """
    logging.debug("Genre Worker: 장르 분류 요청 수신.")
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['audio']

    filename = audio_file.filename.lower()
    temp_input_name = 'temp_input_' + str(os.getpid())
    temp_wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_input_name + '.wav')

    try:
        if filename.endswith('.mp3'):
            temp_mp3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_input_name + '.mp3')
            audio_file.save(temp_mp3_path)
            audio = AudioSegment.from_mp3(temp_mp3_path)
            audio.export(temp_wav_path, format='wav')
            os.remove(temp_mp3_path)
            logging.info(f"MP3 파일 '{filename}'을 WAV로 변환 후 임시 저장: {temp_wav_path}")
        elif filename.endswith('.wav'):
            audio_file.save(temp_wav_path)
            logging.info(f"WAV 파일 '{filename}' 임시 저장: {temp_wav_path}")
        else:
            logging.warning(f"지원하지 않는 파일 형식: {filename}")
            return jsonify({'error': 'Unsupported file format. Please upload mp3 or wav.'}), 400

        features = extract_features_from_audio(temp_wav_path)

        app_cols = list(features.columns)
        if train_cols == app_cols:
            logging.info("피처 컬럼 이름과 순서가 정확히 일치합니다.")
        else:
            logging.warning("피처 컬럼이 일치하지 않습니다!")
            logging.warning(f"학습에 있었으나 현재 없음: {list(set(train_cols) - set(app_cols))}")
            logging.warning(f"현재에 있으나 학습에는 없음: {list(set(app_cols) - set(train_cols))}")
            for i, (tc, ac) in enumerate(zip(train_cols, app_cols)):
                if tc != ac:
                    logging.warning(f"{i}번째 컬럼 불일치: 학습 시 '{tc}' vs 현재 '{ac}'")

        if features['tempo'].dtype == object:
            try:
                features['tempo'] = features['tempo'].astype(float)
                logging.info("'tempo' 컬럼을 float 타입으로 변환했습니다.")
            except Exception as e:
                logging.error(f"'tempo'를 float으로 변환하는 데 실패했습니다: {e}")

        logging.debug(f"학습 시 사용된 피처 개수: {len(train_cols)}")
        logging.debug(f"현재 추출된 피처 개수: {len(app_cols)}")
        logging.debug(f"현재 피처 데이터 타입:\n{features.dtypes}")
        logging.debug(f"스케일링 전 피처:\n{features}")

        features_scaled = scaler.transform(features)
        interpreter.set_tensor(input_details[0]['index'], features_scaled.astype(input_details[0]['dtype']))
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

        logging.debug(f"모델 예측값:\n{preds}")

        label_idx = np.argmax(preds)
        label = genre_labels[label_idx]

        logging.info(f"예측된 장르: {label}")
        return jsonify({'label': label})

    except Exception as e:
        logging.error(f'장르 예측 중 오류 발생: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'error': f'장르 예측 중 오류 발생: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            logging.info(f"임시 파일 삭제: {temp_wav_path}")
        temp_mp3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_input_name + '.mp3')
        if os.path.exists(temp_mp3_path):
            os.remove(temp_mp3_path)
            logging.info(f"임시 MP3 파일 삭제: {temp_mp3_path}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10001))  # 워커는 다른 포트 사용 (로컬 테스트용)
    logging.debug(f"Genre Worker 로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)