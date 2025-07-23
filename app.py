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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한 설정

# 모델과 도구 불러오기
model = tf.keras.models.load_model('saved_models/model.keras')
model.summary()  # 모델 구조 출력 (터미널에 표시됨)

with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('saved_models/genre_labels.json', 'r') as f:
    genre_labels = json.load(f)

# 피처 추출 함수
def extract_features_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=44100)

    # 오디오 신호 유효성 검사
    if np.isnan(y).any():
        raise ValueError("오디오 신호가 유효하지 않습니다.")
    
    # 로드한 오디오 정보 출력
    print(f"Audio loaded: duration={len(y)/sr:.2f} seconds, sr={sr}")

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
    # templates/index.html 파일을 렌더링해서 보여줍니다.
    return render_template('index.html')

# 예측 요청 처리
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['audio']

    # 파일 확장자 검사
    filename = audio_file.filename.lower()
    temp_input = 'temp_input'  # 확장자 제외 임시파일명
    temp_wav = 'temp.wav'

    # 업로드한 파일 임시 저장
    if filename.endswith('.mp3'):
        temp_mp3 = temp_input + '.mp3'
        audio_file.save(temp_mp3)
        # mp3 -> wav 변환
        audio = AudioSegment.from_mp3(temp_mp3)
        audio.export(temp_wav, format='wav')
        os.remove(temp_mp3)
    elif filename.endswith('.wav'):
        temp_wav = temp_input + '.wav'
        audio_file.save(temp_wav)
    else:
        return jsonify({'error': '지원하지 않는 파일 형식입니다. mp3 또는 wav만 업로드해주세요.'}), 400

    try:
        features = extract_features_from_audio(temp_wav)

        # 학습 때 사용한 피처 컬럼명 리스트를 미리 JSON 파일로 저장했다고 가정
        with open('saved_models/feature_columns.json', 'r') as f:
            train_cols = json.load(f)

        app_cols = list(features.columns)

        # 컬럼명 및 순서 비교
        if train_cols == app_cols:
            print("피처 컬럼 이름과 순서가 정확히 일치합니다.")
        else:
            print("피처 컬럼이 일치하지 않습니다!")
            print("학습에 있었으나 현재 없음:", list(set(train_cols) - set(app_cols)))
            print("현재에 있으나 학습에는 없음:", list(set(app_cols) - set(train_cols)))
            for i, (tc, ac) in enumerate(zip(train_cols, app_cols)):
                if tc != ac:
                    print(f"{i}번째 컬럼 불일치: 학습 시 '{tc}' vs 현재 '{ac}'")

        # tempo 컬럼 숫자형 변환 시도
        if features['tempo'].dtype == object:
            try:
                features['tempo'] = features['tempo'].astype(float)
                print("'tempo' 컬럼을 float 타입으로 변환했습니다.")
            except Exception as e:
                print(f"'tempo'를 float으로 변환하는 데 실패했습니다: {e}")    

        print("학습 시 사용된 피처 개수:", len(train_cols))
        print("현재 추출된 피처 개수:", len(app_cols))
        print("현재 피처 데이터 타입:\n", features.dtypes)

        print("\n==== [디버그] 스케일링 전 피처 ====\n", features)  # ✅ 1단계
        features_scaled = scaler.transform(features)
        print("\n==== [디버그] 스케일링 후 피처 ====\n", features_scaled)  # ✅ 2단계
        preds = model.predict(features_scaled)
        print("\n==== [디버그] 모델 예측값 ====\n", preds)  # ✅ 3단계


        print("Scaler mean:\n", scaler.mean_)
        print("Scaler scale:\n", scaler.scale_)
        # features는 extract_features_from_audio()의 결과
        print("Without scaling:")
        raw_preds = model.predict(features)
        print("Model prediction (no scaling):", raw_preds)

        label_idx = np.argmax(preds)
        label = genre_labels[label_idx]

        os.remove(temp_wav)

        return jsonify({'label': label})
    except Exception as e:
        traceback.print_exc()
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return jsonify({'error': f'예측 중 오류 발생: {str(e)}'}), 500
    
if __name__ == '__main__':
    app.run(debug=True)