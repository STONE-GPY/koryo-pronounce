from g2pk import G2p

class KoryoG2PEngine:
    """고려인 대상 한국어 음소 변환 엔진"""
    def __init__(self):
        # g2pk 인스턴스 초기화
        self.g2p = G2p()

    def convert(self, text: str) -> str:
        """문장을 표준 발음 음소열로 변환"""
        if not text:
            return ""
        return self.g2p(text)

    def get_phonemes(self, text: str) -> list:
        """음소 단위 분리가 필요할 때를 대비한 확장 (자소 분리 등)"""
        # 임시로 단어별 리스트 반환
        return self.convert(text).split()
