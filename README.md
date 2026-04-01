# 발음 고려 (Pronunciation Koryo)

고려인 및 한국어 학습자를 위한 인공지능 기반 발음 교정 및 분석 시스템입니다. 음향학적 수치(Formant, VOT)를 기반으로 사용자 발음의 정확도를 측정하고 피드백을 제공합니다.

## 주요 기능
- **G2P (Grapheme-to-Phoneme):** 텍스트를 실제 발음 음소열로 변환 (공백 및 특수 처리 지원)
- **음향 분석 (Acoustic Analysis):** Parselmouth(Praat) 및 Librosa를 활용한 포먼트(F1, F2), 성도 전이, VOT 측정
- **발음 채점:** 표준 발음 수치와 사용자 발음을 비교하여 점수 산출 및 해부학적 피드백 생성
- **개인화 스케일링:** 사용자의 피치(Pitch)에 따른 모음 공간 자동 최적화

## 시작하기

### 1. 환경 설정
Python 3.10 이상의 환경이 필요합니다.
```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 사용 방법 (Step-by-Step)

#### API 서버 실행
웹 브라우저에서 UI를 확인하거나 API를 호출하려면 서버를 실행합니다.
```bash
# FastAPI 서버 실행
uvicorn api:app --host 0.0.0.0 --port 8000
```
- 접속 주소: `http://localhost:8000` (UI 포함)

#### CLI 분석 실행
명령줄에서 특정 오디오 파일을 직접 분석할 수 있습니다.
```bash
# 사용법: python app.py <오디오_경로> <목표_문장>
python app.py data/sample.wav "안녕하세요"
```

#### 테스트 실행
시스템의 안정성을 확인하려면 단위 테스트를 실행합니다.
```bash
python3 -m pytest tests/
```

## 프로젝트 구조
- `src/`: 핵심 분석 로직 (G2P, 음향 분석, 채점기)
- `api.py`: FastAPI 기반 웹 인터페이스
- `app.py`: 통합 분석 애플리케이션 (CLI 지원)
- `static/`: 웹 UI (HTML/CSS/JS)
- `tests/`: 단위 및 통합 테스트 코드
- `data/`: 오디오 샘플 및 임시 업로드 저장소

## 라이선스
본 프로젝트는 교육 및 연구 목적으로 제작되었습니다.
