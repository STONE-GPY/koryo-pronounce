from g2pk import G2p

class KoryoG2P:
    """고려인 대상 한국어 음소 변환 엔진"""
    def __init__(self):
        # g2pk 인스턴스 초기화 (처음 실행 시 데이터 로딩으로 인해 약간의 지연이 있을 수 있음)
        self.g2p = G2p()

    def convert(self, text: str) -> str:
        """문장을 표준 발음 음소열로 변환"""
        # g2pk는 기본적으로 표준 발음 결과를 반환함
        return self.g2p(text)

    def get_phonemes(self, text: str):
        """음소 단위 분리가 필요할 때를 대비한 확장 (자소 분리 등)"""
        # 향후 MFA(Montreal Forced Aligner)와 연동할 때 활용 예정
        pass
