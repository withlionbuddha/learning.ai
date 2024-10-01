from konlpy.tag import Okt
from Lemmatization import Lemmatization

class KoreanLemmatization(Lemmatization):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.okt = Okt()

    def apply_lemmatization(self):
        # 한국어에서는 표제어 추출 대신 어간 추출을 통해 기본형을 유사하게 추출
        # konlpy의 normalize와 stem 사용
        return [self.okt.normalize(token) for token in self.tokens]
