# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import logging
import traceback
import sys
import threading # For unique temporary file names
import time # For timing logging
import json

# Flask 앱 인스턴스 생성 시 static_folder와 template_folder를 명시적으로 설정
app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한 설정

# 로깅 설정: INFO 레벨로 간결하게 출력
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Music Genre App 초기화 시작.")

# 모델 및 도구 파일의 기본 디렉토리
MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 app.py가 있는 디렉토리

# 전역 변수로 선언하여 지연 로딩 및 상태 관리
model = None # Keras 모델
scaler = None
genre_labels = None
train_cols = None
model_loaded = False # 모델 로딩 완료 여부를 추적하는 플래그

def _load_genre_classification_models():
    """
    음악 장르 분류 모델과 관련 도구들을 지연 로드하는 내부 함수.
    """
    global model, scaler, genre_labels, train_cols, model_loaded
    
    if model_loaded: # 이미 로드되었다면 다시 로드하지 않음
        logging.info("모델 및 도구가 이미 로드되어 있습니다. 다시 로드하지 않습니다.")
        return

    logging.info("음악 장르 분류 모델 및 도구 지연 로드 시작.")
    try:
        import tensorflow as tf # Keras 모델 로드를 위해 tensorflow 임포트
        import librosa
        import pandas as pd
        import numpy as np
        import pickle
        
        # 모든 모델/도구 파일 경로 수정: music_genre_app 디렉토리 바로 아래로 설정
        keras_model_path = os.path.join(MODEL_DIR, 'model.keras')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        genre_labels_path = os.path.join(MODEL_DIR, 'genre_labels.json')
        feature_columns_path = os.path.join(MODEL_DIR, 'feature_columns.json')

        logging.info(f"Keras 모델 로드 시도 중: {keras_model_path}")
        model = tf.keras.models.load_model(keras_model_path)
        logging.info("Keras 모델 로드 완료.")

        logging.info(f"스케일러 로드 시도 중: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logging.info("스케일러 로드 완료.")

        logging.info(f"장르 레이블 로드 시도 중: {genre_labels_path}")
        with open(genre_labels_path, 'r') as f:
            genre_labels = json.load(f)
        logging.info("장르 레이블 로드 완료.")
        
        logging.info(f"피처 컬럼 로드 시도 중: {feature_columns_path}")
        with open(feature_columns_path, 'r') as f:
            train_cols = json.load(f)
        logging.info("피처 컬럼 로드 완료.")

        model_loaded = True # 모델 로딩 성공 플래그 설정
        logging.info("음악 장르 분류 Keras 모델 및 도구 로드 성공.")
    except Exception as e:
        logging.error(f"음악 장르 분류 모델 로드 중 오류 발생: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1) 

# 앱 시작 시 모델을 한 번만 로드하도록 설정
with app.app_context():
    _load_genre_classification_models()


def extract_features_from_audio(file_path):
    import librosa
    import numpy as np
    import pandas as pd
    
    try:
        logging.info(f"오디오 파일 로드 시도: {file_path}")
        start_time = time.time()
        y, sr = librosa.load(file_path, sr=44100) # 샘플링 레이트 유지
        end_time = time.time()
        logging.info(f"Audio loaded: duration={len(y)/sr:.2f} seconds, sr={sr}. Load time: {end_time - start_time:.2f}s")

        # 오디오를 30초로 자르는 로직으로 변경
        max_samples = int(30 * sr) # 30초에 해당하는 샘플 수 계산
        if len(y) > max_samples:
            y = y[:max_samples] # 앞부분 30초만 사용
            logging.info(f"오디오 파일이 30초를 초과하여 앞부분 30초로 잘랐습니다. 새 길이: {len(y)/sr:.2f}s")

        if np.isnan(y).any():
            raise ValueError("오디오 신호가 유효하지 않습니다. NaN 값이 포함되어 있습니다.")
        
        # ... (이하 특징 추출 로직 유지) ...
        features = {}

        start_feature_time = time.time()
        logging.info("크로마 STFT 특징 추출 시작...")
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_var'] = np.var(chroma_stft)
        logging.info(f"크로마 STFT 특징 추출 완료. ({time.time() - start_feature_time:.4f}s)")

        start_feature_time = time.time()
        logging.info("RMS 특징 추출 시작...")
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)
        logging.info(f"RMS 특징 추출 완료. ({time.time() - start_feature_time:.4f}s)")

        start_feature_time = time.time()
        logging.info("스펙트럼 센트로이드 특징 추출 시작...")
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spec_centroid)
        features['spectral_centroid_var'] = np.var(spec_centroid)
        logging.info(f"스펙트럼 센트로이드 특징 추출 완료. ({time.time() - start_feature_time:.4f}s)")

        start_feature_time = time.time()
        logging.info("스펙트럼 대역폭 특징 추출 시작...")
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spec_bandwidth)
        features['spectral_bandwidth_var'] = np.var(spec_bandwidth)
        logging.info(f"스펙트럼 대역폭 특징 추출 완료. ({time.time() - start_feature_time:.4f}s)")

        start_feature_time = time.time()
        logging.info("롤오프 특징 추출 시작...")
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_var'] = np.var(rolloff)
        logging.info(f"롤오프 특징 추출 완료. ({time.time() - start_feature_time:.4f}s)")

        start_feature_time = time.time()
        logging.info("제로 크로싱 레이트 특징 추출 시작...")
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_var'] = np.var(zcr)
        logging.info(f"제로 크로싱 레이트 특징 추출 완료. ({time.time() - start_feature_time:.4f}s)")

        start_feature_time = time.time()
        logging.info("하모니 특징 추출 시작...")
        harmony = librosa.effects.harmonic(y)
        features['harmony_mean'] = np.mean(harmony)
        features['harmony_var'] = np.var(harmony)
        logging.info(f"하모니 특징 추출 완료. ({time.time() - start_feature_time:.4f}s)")

        start_feature_time = time.time()
        logging.info("퍼셉트럴 특징 추출 시작...")
        perceptr = librosa.feature.spectral_flatness(y=y)
        features['perceptr_mean'] = np.mean(perceptr)
        features['perceptr_var'] = np.var(perceptr)
        logging.info(f"퍼셉트럴 특징 추출 완료. ({time.time() - start_feature_time:.4f}s)")
        
        start_feature_time = time.time()
        logging.info("템포 특징 추출 시작...")
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        logging.info(f"템포 추출 완료: {tempo}. ({time.time() - start_feature_time:.4f}s)")
        
        start_feature_time = time.time()
        logging.info("MFCC 특징 추출 시작...")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_var'] = np.var(mfcc[i])
        logging.info(f"MFCC 특징 추출 완료. ({time.time() - start_feature_time:.4f}s)")

        logging.info("모든 특징 추출 완료. DataFrame 생성 중...")
        start_df_time = time.time()
        features_df = pd.DataFrame([features])
        logging.info(f"DataFrame 생성 완료. ({time.time() - start_df_time:.4f}s)")

        return features_df
    except Exception as e:
        logging.error(f"오디오 특징 추출 중 오류 발생: {e}")
        logging.error(traceback.format_exc())
        raise # 특징 추출 실패 시 예외를 다시 발생시킴

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/healthz')
def health_check():
    """Render 헬스 체크를 위한 엔드포인트."""
    logging.info("Health check 요청 수신.")
    # 모델 로딩 상태를 헬스 체크에 포함
    if model_loaded:
        return "OK", 200
    else:
        logging.warning("Health check: 모델이 아직 로드되지 않았습니다.")
        return "Model not loaded", 503 # Service Unavailable

@app.route('/predict', methods=['POST'])
def predict_genre_endpoint():
    """
    오디오 파일을 입력받아 장르를 예측하는 엔드포인트입니다.
    """
    if not model_loaded:
        logging.error("장르 예측 요청: 모델이 아직 로드되지 않았습니다. 잠시 후 다시 시도해주세요.")
        return jsonify({'error': 'Model not loaded yet. Please try again in a moment.'}), 503

    if 'audio' not in request.files:
        logging.warning("파일이 업로드되지 않았습니다.")
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['audio']

    filename = audio_file.filename.lower()
    # 고유한 임시 파일명 생성 (여러 요청 동시 처리 시 충돌 방지)
    temp_input_name = 'temp_input_' + str(os.getpid()) + '_' + str(threading.get_ident()) 
    temp_wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_input_name + '.wav')

    try:
        # pydub 임포트 제거 (MP3 지원 안 함)
        import sklearn.preprocessing
        import pandas as pd
        import numpy as np

        logging.info(f"업로드된 파일: {filename}")
        file_processing_start_time = time.time()

        # MP3 처리 로직 제거 (WAV만 허용)
        if filename.endswith('.wav'):
            audio_file.save(temp_wav_path)
            logging.info(f"WAV 파일 '{filename}' 임시 저장: {temp_wav_path}")
            logging.info(f"WAV 파일 저장 완료. ({time.time() - file_processing_start_time:.2f}s)")
        else:
            logging.warning(f"지원하지 않는 파일 형식: {filename}")
            return jsonify({'error': 'Unsupported file format. Please upload only WAV files.'}), 400

        logging.info("오디오 특징 추출 시작...")
        features_extraction_start_time = time.time()
        features_df = extract_features_from_audio(temp_wav_path) # DataFrame으로 반환됨
        logging.info(f"오디오 특징 추출 완료. ({time.time() - features_extraction_start_time:.2f}s)")

        logging.info("피처 데이터프레임 전처리 시작...")
        preprocessing_start_time = time.time()
        # train_cols는 _load_genre_classification_models()에서 로드됨
        processed_features = pd.DataFrame(columns=train_cols)
        for col in train_cols:
            if col in features_df.columns:
                processed_features[col] = features_df[col]
            else:
                processed_features[col] = 0.0 # 없는 컬럼은 0으로 채움
        
        # 모든 컬럼이 숫자형인지 확인하고, 필요한 경우 변환
        for col in processed_features.columns:
            if processed_features[col].dtype == object:
                try:
                    processed_features[col] = pd.to_numeric(processed_features[col], errors='coerce')
                    # 변환 후에도 NaN이 남아있으면 0.0으로 채움
                    processed_features[col] = processed_features[col].fillna(0.0)
                    logging.info(f"'{col}' 컬럼을 숫자 타입으로 변환했습니다.")
                except Exception as e:
                    logging.error(f"'{col}' 컬럼을 숫자 타입으로 변환하는 데 실패했습니다: {e}. 기본값으로 설정합니다.")
                    processed_features[col] = 0.0 # 변환 실패 시 기본값 설정
        logging.info(f"피처 데이터프레임 전처리 완료. ({time.time() - preprocessing_start_time:.2f}s)")

        # 스케일링 전 데이터 로깅
        logging.info(f"스케일링 전 피처 (processed_features):\n{processed_features.head()}")
        logging.info(f"스케일링 전 피처 데이터 타입:\n{processed_features.dtypes}")
        
        # 스케일링
        logging.info("피처 스케일링 시작...")
        scaling_start_time = time.time()
        features_scaled = scaler.transform(processed_features)
        logging.info(f"피처 스케일링 완료. ({time.time() - scaling_start_time:.2f}s)")
        logging.info(f"스케일링 후 피처 (features_scaled):\n{features_scaled}")

        # Keras 모델 예측
        logging.info("Keras 모델 예측 시작...")
        inference_start_time = time.time()
        preds = model.predict(features_scaled) # Keras 모델 예측
        logging.info(f"Keras 모델 예측 완료. ({time.time() - inference_start_time:.2f}s)")

        logging.info(f"모델 예측값:\n{preds}")

        label_idx = np.argmax(preds)
        label = genre_labels[label_idx]
        probability = float(preds[0][label_idx]) * 100 # 확률 계산

        logging.info(f"예측된 장르: {label}, 확률: {probability:.2f}%")
        
        # JSON 응답으로 변경 (현재 앱의 흐름에 맞게)
        return jsonify({'label': label, 'probability': f"{probability:.2f}"}), 200

    except Exception as e:
        logging.error(f'장르 예측 중 오류 발생: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'error': f'장르 예측 중 오류 발생: {str(e)}'}), 500
    finally:
        # 임시 파일 정리
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            logging.info(f"임시 WAV 파일 삭제: {temp_wav_path}")


# 장르별 상세 정보를 담은 딕셔너리 (기존 유지)
GENRE_DETAILS = {
    "blues": {
        "title": "블루스 (Blues)",
        "origin": "19세기 말 미국 남부 아프리카계 미국인 공동체에서 시작된 음악 장르입니다.",
        "features": "주요 특징은 블루 노트, 콜 앤 리스폰스(Call and Response) 형식, 서정적이고 감성적인 가사입니다.",
        "bpm_range": "BPM은 대개 60-90 사이로 느린 편입니다."
    },
    "classical": {
        "title": "클래식 (Classical)",
        "origin": "서양 예술 음악의 한 장르로, 주로 18세기 중반부터 19세기 초까지의 시대를 지칭합니다.",
        "features": "정교한 형식과 구조, 조화로운 선율과 화성, 풍부한 오케스트레이션이 특징입니다.",
        "bpm_range": "BPM은 곡에 따라 매우 다양하나, 대체로 템포 변화가 큽니다."
    },
    "country": {
        "title": "컨트리 (Country)",
        "origin": "1920년대 미국 남부의 민속 음악과 서부 카우보이 음악에서 유래했습니다.",
        "features": "단순하고 서정적인 멜로디, 어쿠스틱 기타, 밴조, 피들 등의 악기 사용, 일상생활과 자연에 대한 가사가 특징입니다.",
        "bpm_range": "BPM은 대개 80-120 사이입니다."
    },
    "disco": {
        "title": "디스코 (Disco)",
        "origin": "1970년대 뉴욕 클럽 문화에서 시작된 댄스 음악 장르입니다.",
        "features": "강렬한 4/4박자 비트, 신시사이저, 스트링, 혼 섹션 사용, 춤추기 좋은 리듬이 특징입니다.",
        "bpm_range": "BPM은 대개 110-130 사이입니다."
    },
    "hiphop": {
        "title": "힙합 (Hip-Hop)",
        "origin": "1970년대 뉴욕 브롱스에서 시작된 문화 운동의 일부입니다.",
        "features": "랩(Rapping), 디제잉(DJing), 샘플링, 강력한 비트와 리듬, 사회 비판적 또는 자전적 가사가 특징입니다.",
        "bpm_range": "BPM은 대개 80-120 사이입니다."
    },
    "jazz": {
        "title": "재즈 (Jazz)",
        "origin": "19세기 말 미국 뉴올리언스의 흑인 문화에서 비롯되었습니다.",
        "features": "즉흥 연주, 스윙 리듬, 4/4 박자, 고유한 코드 진행, 다양한 악기들의 상호작용이 특징입니다.",
        "bpm_range": "BPM은 대개 128~185 사이로 다양합니다."
    },
    "metal": {
        "title": "메탈 (Metal)",
        "origin": "1960년대 후반 하드 록에서 파생된 장르입니다.",
        "features": "강력한 기타 리프, 왜곡된 사운드, 빠른 드럼 비트, 공격적인 보컬, 복잡한 곡 구조가 특징입니다.",
        "bpm_range": "BPM은 대개 120-200 이상으로 매우 빠를 수 있습니다."
    },
    "pop": {
        "title": "팝 (Pop)",
        "origin": "대중적인 인기를 얻기 위해 만들어진 상업적인 음악 장르입니다.",
        "features": "쉽고 따라 부르기 쉬운 멜로디, 다양한 스타일의 혼합, 최신 트렌드 반영, 반복적인 후렴구가 특징입니다.",
        "bpm_range": "BPM은 대개 90-140 사이입니다."
    },
    "reggae": {
        "title": "레게 (Reggae)",
        "origin": "1960년대 후반 자메이카에서 시작된 음악 장르입니다.",
        "features": "오프비트(off-beat) 리듬, 베이스 기타와 드럼의 강조, 종교적 또는 사회적 메시지를 담은 가사가 특징입니다.",
        "bpm_range": "BPM은 대개 60-90 사이로 느린 편입니다."
    },
    "rock": {
        "title": "록 (Rock)",
        "origin": "1950년대 미국에서 로큰롤에서 파생된 장르입니다.",
        "features": "강렬한 기타 리프, 드럼 비트, 보컬이 특징이며, 다양한 하위 장르를 가집니다. 반항적이고 자유로운 정신을 표현합니다.",
        "bpm_range": "BPM은 대개 100-180 사이입니다."
    }
}

@app.route('/result')
def show_genre_result():
    """장르 분석 결과 페이지 렌더링."""
    genre_key = request.args.get('genre', '기타').lower()
    probability = request.args.get('probability', '0%')

    genre_info = GENRE_DETAILS.get(genre_key, {
        "title": "기타 (Other)",
        "origin": "이 음악은 특정 장르로 분류하기 어렵거나, 새로운 장르일 수 있습니다.",
        "features": "특징적인 요소들을 파악하기 어렵습니다.",
        "bpm_range": "BPM 정보는 알 수 없습니다."
    })

    return render_template('genre_result_page.html', 
                           genre=genre_key, # JavaScript에서 사용하기 위해 전달
                           genre_display_name=genre_info["title"],
                           probability=probability,
                           genre_origin=genre_info["origin"],
                           genre_features=genre_info["features"],
                           genre_bpm_range=genre_info["bpm_range"])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Music Genre App 로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)