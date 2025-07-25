from flask import Flask, request, jsonify, render_template
import os
import librosa
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
from pydub import AudioSegment
import logging
import traceback
import sys

# --- 음악 추천 기능 관련 모듈 임포트 ---
# music-recommendation-by-bpm 디렉토리를 Python 경로에 추가하여 패키지 임포트가 가능하도록 합니다.
# app.py가 프로젝트 루트에 있으므로, music-recommendation-by-bpm을 직접 임포트합니다.
project_root = os.path.dirname(os.path.abspath(__file__))
music_rec_dir = os.path.join(project_root, 'music-recommendation-by-bpm')

if music_rec_dir not in sys.path:
    sys.path.append(music_rec_dir)

try:
    # 'music-recommendation-by-bpm'이 패키지로 인식되므로,
    # 해당 패키지 내의 music_recommender 모듈을 임포트합니다.
    from music_recommender import MusicRecommender
except ImportError as e:
    logging.error(f"오류: 'music_recommender' 모듈을 임포트할 수 없습니다. {e}")
    logging.error("다음 사항을 확인해주세요:")
    logging.error("1. 'music-recommendation-by-bpm' 디렉토리 안에 '__init__.py' 파일이 있는지 확인.")
    logging.error("2. 'music_recommender.py' 파일이 'music-recommendation-by-bpm' 디렉토리 안에 있는지 확인.")
    logging.error("3. 가상 환경이 활성화되어 있고 필요한 라이브러리(transformers, torch, requests)가 설치되었는지 확인.")
    sys.exit(1) # 모듈 로드 실패 시 애플리케이션 종료

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한 설정

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 음악 장르 분류 기능 관련 모델 및 도구 로드 ---
# 모델 파일 경로를 절대 경로 또는 app.py 기준으로 상대 경로로 정확히 지정해야 합니다.
# 여기서는 app.py가 프로젝트 루트에 있다고 가정하고 상대 경로를 사용합니다.
MODEL_DIR = os.path.join(project_root, 'music-genre-classification', 'saved_models')
# TEMPLATES_DIR은 이제 필요 없습니다. Flask가 최상위 templates 폴더를 자동으로 찾게 설정합니다.
# TEMPLATES_DIR = os.path.join(project_root, 'music-genre-classification', 'templates') # 이 줄은 삭제하거나 주석 처리

# Flask 앱의 템플릿 폴더를 최상위 'templates' 폴더로 설정
app.template_folder = os.path.join(project_root, 'templates') # 이 줄을 추가

try:
    model_path = os.path.join(MODEL_DIR, 'model.keras')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    genre_labels_path = os.path.join(MODEL_DIR, 'genre_labels.json')
    feature_columns_path = os.path.join(MODEL_DIR, 'feature_columns.json')

    model = tf.keras.models.load_model(model_path)
    model.summary()  # 모델 구조 출력 (터미널에 표시됨)
    # model.save('model.keras', save_format='keras') # 이 줄은 모델을 다시 저장하는 것이므로, 로드 시에는 필요 없습니다.

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    with open(genre_labels_path, 'r') as f:
        genre_labels = json.load(f)
    
    with open(feature_columns_path, 'r') as f:
        train_cols = json.load(f)

    logging.info("음악 장르 분류 모델 및 도구 로드 성공.")
except Exception as e:
    logging.error(f"음악 장르 분류 모델 로드 중 오류 발생: {e}")
    logging.error(traceback.format_exc())
    sys.exit(1) # 모델 로드 실패 시 애플리케이션 종료

# --- 2. 음악 추천 기능 관련 MusicRecommender 인스턴스 초기화 ---
# getsongbpm API 키는 환경 변수로 관리하는 것이 보안상 가장 안전합니다.
# 환경 변수 GETSONGBPM_API_KEY가 설정되어 있지 않으면
# music_recommender.py 내부에서 Mock 데이터를 사용하게 됩니다.
getsongbpm_api_key = os.environ.get("GETSONGBPM_API_KEY", "YOUR_GETSONGBPM_API_KEY_HERE")

# MusicRecommender 인스턴스 생성
# API 키가 'YOUR_GETSONGBPM_API_KEY_HERE'이거나 환경 변수에 없으면
# music_recommender.py 내부 로직에 따라 Mock Recommender가 사용됩니다.
recommender = MusicRecommender(getsongbpm_api_key)
logging.info("음악 추천 시스템 인스턴스 초기화 완료.")


# --- 1. 음악 장르 분류 기능 엔드포인트 ---

# 피처 추출 함수 (기존 코드와 동일)
def extract_features_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=44100)

    # 오디오 신호 유효성 검사
    if np.isnan(y).any():
        raise ValueError("오디오 신호가 유효하지 않습니다.")
    
    # 로드한 오디오 정보 출력
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
    # 이제 Flask 앱의 template_folder가 'MUSIC_INTELLIGENCE/templates'로 설정되었으므로,
    # 'index.html'만 지정하면 됩니다.
    return render_template('index.html') # 이 줄만 변경 (이전 os.path.join(TEMPLATES_DIR, 'index.html') 에서)


# 예측 요청 처리 (기존 코드와 동일)
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        logging.warning("파일이 업로드되지 않았습니다.")
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['audio']

    filename = audio_file.filename.lower()
    temp_input_name = 'temp_input_' + str(os.getpid()) # 프로세스 ID를 추가하여 파일명 충돌 방지
    temp_wav_path = os.path.join(project_root, temp_input_name + '.wav') # 프로젝트 루트에 임시 파일 저장

    try:
        # 업로드한 파일 임시 저장 및 wav 변환
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

        # 피처 컬럼명 및 순서 비교
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

        # tempo 컬럼 숫자형 변환 시도
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

        features_scaled = scaler.transform(features[train_cols]) # 학습 시 사용된 컬럼 순서로 변환
        logging.debug(f"스케일링 후 피처:\n{features_scaled}")

        preds = model.predict(features_scaled)
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
        # 임시 파일 정리
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            logging.info(f"임시 파일 삭제: {temp_wav_path}")
        # mp3 임시 파일도 혹시 남아있을 경우 대비
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
    data = request.get_json()
    user_text = data.get('text')

    if not user_text:
        logging.warning("추천 요청에 텍스트가 없습니다.")
        return jsonify({'error': '텍스트를 입력해주세요.'}), 400

    try:
        # MusicRecommender의 recommend_music 메서드 호출
        # 이 메서드는 이미 내부적으로 감정 분석, BPM 매핑, 노래 검색(Mock)을 수행하고
        # 콘솔에 결과를 출력합니다.
        recommended_songs = recommender.recommend_music(user_text)

        # 웹 응답을 위해 추천된 노래 리스트만 JSON으로 반환합니다.
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
    # Flask 서버 실행
    # debug=True는 개발 중에만 사용하고, 실제 배포 시에는 False로 설정하거나 gunicorn과 같은 WSGI 서버를 사용해야 합니다.
    app.run(debug=True, host='0.0.0.0', port=5000)