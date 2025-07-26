from flask import Flask, request, jsonify, render_template
import os
import logging
import traceback
import sys
# librosa, pandas, numpy, tensorflow, pickle, json, pydub 임포트는 이제 함수 내부로 이동합니다.

# --- 음악 추천 기능 관련 모듈 임포트 ---
project_root = os.path.dirname(os.path.abspath(__file__))
music_rec_dir = os.path.join(project_root, 'music-recommendation-by-bpm')

logging.debug(f"프로젝트 루트: {project_root}")
logging.debug(f"음악 추천 모듈 디렉토리: {music_rec_dir}")

if music_rec_dir not in sys.path:
    sys.path.append(music_rec_dir)
    logging.debug(f"'{music_rec_dir}'를 sys.path에 추가했습니다.")

# MusicRecommender 모듈 자체는 여전히 앱 시작 시 임포트됩니다.
# MusicRecommender 내부의 SentimentAnalyzer는 이미 지연 로딩됩니다.
try:
    logging.debug("music_recommender 모듈 임포트 시도 중...")
    from music_recommender import MusicRecommender
    logging.debug("music_recommender 모듈 임포트 성공.")
except ImportError as e:
    logging.error(f"오류: 'music_recommender' 모듈을 임포트할 수 없습니다. {e}")
    logging.error("다음 사항을 확인해주세요:")
    logging.error("1. 'music-recommendation-by-bpm' 디렉토리 안에 '__init__.py' 파일이 있는지 확인.")
    logging.error("2. 'music_recommender.py' 파일이 'music-recommendation-by-bpm' 디렉토리 안에 있는지 확인.")
    logging.error("3. 가상 환경이 활성화되어 있고 필요한 라이브러리(transformers, torch, requests)가 설치되었는지 확인.")
    sys.exit(1)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한 설정

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Flask 앱 초기화 시작.")

# --- 1. 음악 장르 분류 기능 관련 모델 및 도구 로드 (지연 로딩을 위해 전역 변수로 선언) ---
MODEL_DIR = os.path.join(project_root, 'music-genre-classification', 'saved_models')
app.template_folder = os.path.join(project_root, 'templates')

# 모델 관련 변수들을 None으로 초기화하여 지연 로딩을 준비
# 이 변수들은 이제 _load_genre_classification_models 함수 내에서 global로 선언됩니다.
interpreter = None
input_details = None
output_details = None
scaler = None
genre_labels = None
train_cols = None

def _load_genre_classification_models():
    """
    음악 장르 분류 모델과 관련 도구들을 지연 로드하는 내부 함수.
    이 함수가 호출될 때 관련 라이브러리(tensorflow, librosa, pandas, numpy, pickle, json, pydub)를 임포트합니다.
    """
    global interpreter, input_details, output_details, scaler, genre_labels, train_cols
    
    # 필요한 라이브러리들을 함수 내부에서 임포트 (극단적인 지연 로딩)
    import tensorflow.lite as tflite
    import librosa
    import pandas as pd
    import numpy as np
    import pickle
    import json
    # from pydub import AudioSegment # pydub은 predict 함수 내에서만 필요하므로 여기서는 제거

    if interpreter is not None: # 이미 로드되었다면 다시 로드하지 않음
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

        logging.debug(f"스케일러 로드 시도 중 (지연 로드): {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logging.debug("스케일러 로드 완료 (지연 로드).")

        logging.debug(f"장르 레이블 로드 시도 중 (지연 로드): {genre_labels_path}")
        with open(genre_labels_path, 'r') as f:
            genre_labels = json.load(f)
        logging.debug("장르 레이블 로드 완료 (지연 로드).")
        
        logging.debug(f"피처 컬럼 로드 시도 중 (지연 로드): {feature_columns_path}")
        with open(feature_columns_path, 'r') as f:
            train_cols = json.load(f)
        logging.debug("피처 컬럼 로드 완료 (지연 로드).")

        logging.info("음악 장르 분류 TFLite 모델 및 도구 지연 로드 성공.")
    except Exception as e:
        logging.error(f"음악 장르 분류 모델 지연 로드 중 오류 발생: {e}")
        logging.error(traceback.format_exc())
        raise # 예외를 다시 발생시켜 상위 호출자에게 알림

# extract_features_from_audio 함수 자체에 필요한 라이브러리 임포트 추가
def extract_features_from_audio(file_path):
    import librosa
    import numpy as np # <-- np 경고 해결: 함수 내에서 numpy 임포트
    import pandas as pd

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 음악 장르 분류 모델을 사용하기 전에 지연 로드
    _load_genre_classification_models()

    if 'audio' not in request.files:
        logging.warning("파일이 업로드되지 않았습니다.")
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['audio']

    filename = audio_file.filename.lower()
    temp_input_name = 'temp_input_' + str(os.getpid())
    temp_wav_path = os.path.join(project_root, temp_input_name + '.wav')

    try:
        from pydub import AudioSegment # pydub 임포트도 이 함수 내부로 이동
        # scikit-learn 임포트도 이 함수 내부로 이동 (scaler.transform 사용 시 필요)
        import sklearn.preprocessing # StandardScaler가 이 모듈에 있습니다.

        if filename.endswith('.mp3'):
            temp_mp3_path = os.path.join(project_root, temp_input_name + '.mp3')
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
            return jsonify({'error': '지원하지 않는 파일 형식입니다. mp3 또는 wav만 업로드해주세요.'}), 400

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

        features_scaled = scaler.transform(features[train_cols])
        logging.debug(f"스케일링 후 피처:\n{features_scaled}")

        # TFLite 모델을 사용하여 예측 수행
        interpreter.set_tensor(input_details[0]['index'], features_scaled.astype(input_details[0]['dtype']))
        interpreter.invoke() # 추론 실행
        preds = interpreter.get_tensor(output_details[0]['index']) # 결과 가져오기

        logging.debug(f"모델 예측값:\n{preds}")

        label_idx = np.argmax(preds)
        label = genre_labels[label_idx]

        logging.info(f"예측된 장르: {label}")
        return jsonify({'label': label})

    except Exception as e:
        logging.error(f'예측 중 오류 발생: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'error': f'예측 중 오류 발생: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            logging.info(f"임시 파일 삭제: {temp_wav_path}")
        temp_mp3_path = os.path.join(project_root, temp_input_name + '.mp3')
        if os.path.exists(temp_mp3_path):
            os.remove(temp_mp3_path)
            logging.info(f"임시 MP3 파일 삭제: {temp_mp3_path}")


# --- 2. 음악 추천 기능 엔드포인트 ---
@app.route('/recommend', methods=['POST'])
def recommend_music_endpoint():
    """
    사용자 텍스트를 입력받아 감정 기반 음악을 추천하는 엔드포인트입니다.
    """
    # recommender는 전역 변수로 이미 정의되어 있으므로, Pylance 경고는 무시해도 되지만
    # 명시적으로 global 선언을 추가하여 경고를 없앨 수도 있습니다.
    # global recommender # <-- 이 줄을 추가하면 Pylance 경고가 사라질 수 있습니다.

    data = request.get_json()
    user_text = data.get('text')

    if not user_text:
        logging.warning("추천 요청에 텍스트가 없습니다.")
        return jsonify({'error': '텍스트를 입력해주세요.'}), 400

    try:
        recommended_songs = recommender.recommend_music(user_text)

        if recommended_songs:
            logging.info(f"'{user_text}'에 대한 음악 추천 완료. {len(recommended_songs)}곡 추천.")
            return jsonify({'recommendations': recommended_songs}), 200
        else:
            logging.info(f"'{user_text}'에 대한 추천 음악을 찾을 수 없습니다.")
            return jsonify({'message': '추천 음악을 찾을 수 없습니다.'}), 200

    except Exception as e:
        logging.error(f"음악 추천 중 오류 발생: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'음악 추천 중 오류 발생: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.debug(f"로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)