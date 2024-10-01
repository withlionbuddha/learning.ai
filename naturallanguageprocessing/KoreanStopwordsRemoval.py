from StopwordsRemoval import StopwordsRemoval

class KoreanStopwordsRemoval(StopwordsRemoval):
    def __init__(self, tokens):
        super().__init__(tokens)
        # 한국어 불용어 목록
        self.stopwords = ['은', '는', '이', '가', '를', '에', '의', '을', '도', '으로', '하다', '있다', '되다']

    def remove_stopwords(self):
        # 한국어 불용어 제거
        return [token for token in self.tokens if token not in self.stopwords]
