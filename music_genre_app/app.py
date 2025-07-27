from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import logging
import traceback
import sys

# Flask 앱 인스턴스 생성 시 static_folder와 template_folder를 명시적으로 설정
app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한 설정

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Music Genre App 초기화 시작.")

# --- 음악 장르 분류 기능 관련 모델 및 도구 로드 (지연 로딩을 위해 전역 변수로 선언) ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 app.py가 있는 디렉토리

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
    import pandas as pd # pandas 임포트 유지
    import numpy as np # numpy 임포트 유지
    import pickle
    import json
    
    if interpreter is not None: # 이미 로드되었다면 다시 로드하지 않음
        logging.debug("모델 및 도구가 이미 로드되어 있습니다. 다시 로드하지 않습니다.")
        return

    logging.debug("음악 장르 분류 모델 및 도구 지연 로드 시작.")
    try:
        # 모델 파일 경로를 MODEL_DIR (music_genre_app 디렉토리) 바로 아래로 설정
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
        sys.exit(1) # 모델 로드 실패 시 앱 종료 (배포 환경에서 중요)

def extract_features_from_audio(file_path):
    import librosa
    import numpy as np
    import pandas as pd
    
    try:
        y, sr = librosa.load(file_path, sr=44100)

        if np.isnan(y).any():
            raise ValueError("오디오 신호가 유효하지 않습니다. NaN 값이 포함되어 있습니다.")
        
        logging.info(f"Audio loaded: duration={len(y)/sr:.2f} seconds, sr={sr}")

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmony = librosa.effects.harmonic(y)
        perceptr = librosa.feature.spectral_flatness(y=y)
        
        # tempo는 librosa.beat.beat_track이 튜플을 반환하므로 첫 번째 요소만 사용
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
            'tempo': tempo, # tempo는 스칼라 값으로 바로 할당
        }

        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_var'] = np.var(mfcc[i])

        return pd.DataFrame([features])
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
    logging.debug("Health check 요청 수신.")
    return "OK", 200

@app.route('/predict', methods=['POST'])
def predict_genre_endpoint():
    """
    오디오 파일을 입력받아 장르를 예측하는 엔드포인트입니다.
    """
    _load_genre_classification_models() # 모델 로드

    if 'audio' not in request.files:
        logging.warning("파일이 업로드되지 않았습니다.")
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['audio']

    filename = audio_file.filename.lower()
    temp_input_name = 'temp_input_' + str(os.getpid())
    temp_wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_input_name + '.wav')

    try:
        from pydub import AudioSegment
        import sklearn.preprocessing
        import pandas as pd # <-- 여기에 pandas 임포트 추가
        import numpy as np # <-- 여기에 numpy 임포트 추가

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

        features_df = extract_features_from_audio(temp_wav_path) # DataFrame으로 반환됨

        # train_cols와 features_df의 컬럼 순서 및 일치 여부 확인
        # train_cols에 없는 컬럼이 features_df에 있다면 제거
        # features_df에 없는 컬럼이 train_cols에 있다면 0으로 채우기
        processed_features = pd.DataFrame(columns=train_cols)
        for col in train_cols:
            if col in features_df.columns:
                processed_features[col] = features_df[col]
            else:
                processed_features[col] = 0.0 # 없는 컬럼은 0으로 채움

        # tempo 컬럼이 object 타입일 경우 float으로 변환 시도
        if processed_features['tempo'].dtype == object:
            try:
                processed_features['tempo'] = processed_features['tempo'].astype(float)
                logging.info("'tempo' 컬럼을 float 타입으로 변환했습니다.")
            except Exception as e:
                logging.error(f"'tempo'를 float으로 변환하는 데 실패했습니다: {e}. 기본값으로 설정합니다.")
                processed_features['tempo'] = 0.0 # 변환 실패 시 기본값 설정

        # 스케일링 전 데이터 로깅
        logging.debug(f"스케일링 전 피처 (processed_features):\n{processed_features.head()}")
        logging.debug(f"스케일링 전 피처 데이터 타입:\n{processed_features.dtypes}")
        
        # 스케일링
        features_scaled = scaler.transform(processed_features)
        logging.debug(f"스케일링 후 피처 (features_scaled):\n{features_scaled}")

        # TFLite 모델 예측
        interpreter.set_tensor(input_details[0]['index'], features_scaled.astype(input_details[0]['dtype']))
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

        logging.debug(f"모델 예측값:\n{preds}")

        label_idx = np.argmax(preds)
        label = genre_labels[label_idx]
        probability = float(preds[0][label_idx]) * 100 # 확률 계산

        logging.info(f"예측된 장르: {label}, 확률: {probability:.2f}%")
        
        # JSON 응답으로 변경
        return jsonify({'label': label, 'probability': f"{probability:.2f}"}), 200

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

# 장르별 상세 정보를 담은 딕셔너리
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
    genre_key = request.args.get('genre', '기타').lower() # 소문자로 변환하여 키로 사용
    probability = request.args.get('probability', '0%')

    # GENRE_DETAILS 딕셔너리에서 해당 장르 정보 가져오기
    # 키가 없는 경우 '기타' 장르 정보 사용
    genre_info = GENRE_DETAILS.get(genre_key, {
        "title": "기타 (Other)",
        "origin": "이 음악은 특정 장르로 분류하기 어렵거나, 새로운 장르일 수 있습니다.",
        "features": "특징적인 요소들을 파악하기 어렵습니다.",
        "bpm_range": "BPM 정보는 알 수 없습니다."
    })

    return render_template('genre_result_page.html', 
                           genre=genre_key, # 실제 장르 키 (예: 'jazz')
                           genre_display_name=genre_info["title"], # 표시될 장르 이름 (예: '재즈 (Jazz)')
                           probability=probability,
                           genre_origin=genre_info["origin"],
                           genre_features=genre_info["features"],
                           genre_bpm_range=genre_info["bpm_range"])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.debug(f"Music Genre App 로컬 개발 서버 시작 시도 중 (host=0.0.0.0, port={port})...")
    app.run(debug=True, host='0.0.0.0', port=port)