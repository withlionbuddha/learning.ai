from Encoding import Encoding

class KoreanEncoding(Encoding):
    def __init__(self, tokens):
        super().__init__(tokens)
        # 한국어 단어에 대한 간단한 어휘집 생성
        self.vocab = {word: idx for idx, word in enumerate(set(tokens), start=2)}  # 0: <pad>, 1: <unk>

    def encode(self):
        # 한국어 토큰을 어휘집을 사용해 숫자로 변환
        return [self.vocab.get(token, 1) for token in self.tokens]  # 1은 <unk>로 처리
