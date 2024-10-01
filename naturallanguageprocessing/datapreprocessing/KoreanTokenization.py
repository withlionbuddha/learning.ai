
from konlpy.tag import Okt
from Tokenization import Tokenization

class KoreanTokenization(Tokenization):
    def __init__(self, text):
        super().__init__(text)
        self.okt = Okt() # okt instance 생성

    def tokenize(self):
        # 한국어 형태소 기반 토큰화
        return self.okt.morphs(self.text)
