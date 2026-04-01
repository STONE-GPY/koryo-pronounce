class AudioConfig:
    """오디오 공통 설정"""
    SAMPLE_RATE = 16000

class PraatConfig:
    """Praat(Parselmouth) 분석 엔진 파라미터"""
    TIME_STEP = 0.01
    MAX_FORMANTS = 5
    MAX_FORMANT_FREQ = 5500.0
    WINDOW_LENGTH = 0.025
    PRE_EMPHASIS = 50.0

class VotConfig:
    """VOT (파열음) 자동 탐지 설정"""
    DEFAULT_VOT_ON_FAIL = 30.0 # 탐지 실패 시 기본값
    RMS_MULTIPLIER_THRESHOLD = 1.5 # 에너지 상승 임계치
    MAX_VOT_MS = 150.0 # VOT 최대 한계치

class PitchConfig:
    """Pitch (성대 진동) 및 성별 판단 기준"""
    MALE_BASE_PITCH = 120.0
    GENDER_THRESHOLD = 165.0
    MIN_VALID_PITCH = 50.0

class ScoringConfig:
    """점수 산출 및 모음 동적 스케일링 파라미터"""
    SCALE_FACTOR_SLOPE = 0.002 # 10Hz당 상승 비율
    MIN_SCALE_FACTOR = 0.8
    MAX_SCALE_FACTOR = 1.4
    VOWEL_PENALTY_DIVISOR = 15.0 # 모음 거리 감점 가중치
