from flask import Flask, request, jsonify, render_template
import os
import logging
import traceback
import sys

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한 설정

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Music Genre App 초기화 시작.")

# --- 음악 장르 분류 기능 관련 모델 및 도구 로드 (지연 로딩을 위해 전역 변수로 선언) ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
app.template_folder = os.path.join(MODEL_DIR, 'templates')

interpreter = None
input_details = None
output_details = None
scaler = None
genre_labels = None
train_cols = None

def _load_genre_classification_models():
    global interpreter, input_details, output_details, scaler, genre_labels, train_cols

    import tensorflow.lite as tflite
    import librosa
    import pandas as pd
    import numpy as np
    import pickle
    import json

    if interpreter is not None:
        return

    logging.debug("음악 장르 분류 모델 및 도구 지연 로드 시작.")
    try:
        tflite_model_path = os.path.join(MODEL_DIR, 'quantized_model.tflite')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        genre_labels_path = os.path.join(MODEL_DIR, 'genre_labels.json')
        feature_columns_path = os.path.join(MODEL_DIR, 'feature_columns.json')

        logging.debug(f"TFLite 모델 로드 시도 중 (지연 로드): {tflite_model_path}")
        interpreter = tflite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logging.debug("TFLite 모델 로드 및 텐서 할당 완료 (지연 로드).")

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logging.debug("스케일러 로드 완료 (지연 로드).")

        with open(genre_labels_path, 'r') as f:
            genre_labels = json.load(f)
        logging.debug("장르 레이블 로드 완료 (지연 로드).")

        with open(feature_columns_path, 'r') as f:
            train_cols = json.load(f)
        logging.debug("피처 컬럼 로드 완료 (지연 로드).")

        logging.info("음악 장르 분류 TFLite 모델 및 도구 지연 로드 성공.")
    except Exception as e:
        logging.error(f"음악 장르 분류 모델 지연 로드 중 오류 발생: {e}")
        logging.error(traceback.format_exc())
        raise

def extract_features_from_audio(file_path):
    import librosa
    import numpy as np
    import pandas as pd

    y, sr = librosa.load(file_path, sr=44100)

    if np.isnan(y).any():
        raise ValueError("오디오 신호가 유효하지 않습니다.")

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/healthz')
def health_check():
    return "OK", 200

@app.route('/predict', methods=['POST'])
def predict_genre_endpoint():
    _load_genre_classification_models()

    if 'audio' not in request.files:
        logging.warning("파일이 업로드되지 않았습니다.")
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['audio']
    filename = audio_file.filename.lower()
    temp_input_name = 'temp_input_' + str(os.getpid())
    temp_wav_path = os.path.join(MODEL_DIR, temp_input_name + '.wav')

    try:
        from pydub import AudioSegment
        import sklearn.preprocessing

        if filename.endswith('.mp3'):
            temp_mp3_path = os.path.join(MODEL_DIR, temp_input_name + '.mp3')
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

        if train_cols != list(features.columns):
            logging.warning("피처 컬럼이 일치하지 않습니다!")

        if features['tempo'].dtype == object:
            features['tempo'] = features['tempo'].astype(float)

        features_scaled = scaler.transform(features[train_cols])
        interpreter.set_tensor(input_details[0]['index'], features_scaled.astype(input_details[0]['dtype']))
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
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
        temp_mp3_path = os.path.join(MODEL_DIR, temp_input_name + '.mp3')
        if os.path.exists(temp_mp3_path):
            os.remove(temp_mp3_path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.debug(f"Music Genre App 로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)