# koryo-pronounce

고려인 한국어 학습자를 위한 음향 분석 및 음성 인식 기반 발음 교정 시스템입니다.

## 주요 기능
- **Hybrid Evaluation:** WhisperX의 인식 신뢰도(60%)와 Acoustic 엔진의 포먼트/VOT 정밀 분석(40%)을 결합한 채점 로직.
- **Denoising:** 스펙트럼 차감법(Spectral Subtraction) 및 Band-pass Filter(80Hz-8000Hz)를 통한 오디오 전처리.
- **Partial Matching:** 발화가 목표 문장보다 길거나 추가 발화가 포함된 경우에도 포함 관계를 분석하여 정당한 점수 산출.
- **Dialect Support:** 고려말 방언 특성(ㅢ->ㅣ, ㅕ->ㅔ 등) 감지 시 점수 보전 및 피드백 제공.
- **Acoustic Analysis:** Praat(Parselmouth)를 활용한 F1/F2 포먼트 추출 및 파열음(ㄱ,ㄲ,ㅋ) VOT 측정.

## 기술 스택
- **Audio:** librosa, parselmouth, scipy, pydub
- **STT/Alignment:** WhisperX (Faster-Whisper + Pyannote VAD)
- **G2P:** g2pk
- **Server:** FastAPI

## 설치 및 실행

### 1. 환경 설정
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 실행 방법
- **CLI 분석:**
  ```bash
  python app.py <audio_path> <target_text>
  ```
- **API 서버:**
  ```bash
  uvicorn api:app --host 0.0.0.0 --port 8000
  ```

## 데이터셋 구조 (`data/`)
- `anchor/`: 표준어 화자(뉴스 앵커 등)의 기준 음원 및 텍스트.
- `koryo/`: 고려인 인터뷰 및 학습자 발음 데이터.
- `*.wav`: 시스템 검증용 표준 TTS 샘플.

## 핵심 로직 (app.py)
1. `AudioProcessor.denoise`: 오디오 로드 및 배경 소음 제거.
2. `WhisperXProcessor.transcribe_and_align`: 단어 단위 타임스탬프 및 인식률 추출.
3. `PronunciationApp.analyze_hybrid`: WhisperX 결과와 Acoustic 분석 수치 결합.
4. `PronunciationScorer`: 화자 Pitch 기반 포먼트 스케일링 및 고려말 방언 규칙 적용.

## 시스템 평가 방식 직접 비교하기 (Step-by-Step)
정식 애플리케이션 실행 파일인 `app.py`를 통해 두 가지 채점 방식(Acoustic 엔진 단독 분석 vs Hybrid 엔진)의 차이를 직접 테스트해 볼 수 있습니다.

```bash
# 1. 기본 음향(Acoustic) 방식 테스트 (물리적 해부학 교정 위주)
python app.py data/koryo/seg_019_188.2.wav "아이들이 이렇게 오니까 혹시나 문제가 생기지 않을까 싶어서 좀 불안한 부분이 많은데 어쩔 수 없이 여기서 수업을 진행할 수 밖에 없"

# 2. 하이브리드(Hybrid) 방식 테스트 (AI 인식률 결합 및 방언 수용, `--hybrid` 플래그 추가)
python app.py data/koryo/seg_019_188.2.wav "아이들이 이렇게 오니까 혹시나 문제가 생기지 않을까 싶어서 좀 불안한 부분이 많은데 어쩔 수 없이 여기서 수업을 진행할 수 밖에 없" --hybrid
```

**결과 확인 포인트:**
- **Acoustic (`app.py` 기본):** 모든 음절의 물리적 주파수(F1/F2) 위치를 엄격하게 측정하여 점수가 낮게 나오더라도 세밀한 입 모양 교정 피드백을 제공합니다.
- **Hybrid (`--hybrid` 플래그):** 실제 대화 환경을 고려하여 WhisperX의 AI 인식률을 반영하고, 부분 일치 로직을 통해 부당한 감점 없이 전달력을 평가합니다.
