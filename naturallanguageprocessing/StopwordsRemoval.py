from abc import ABC, abstractmethod

class StopwordsRemoval(ABC):
    def __init__(self, tokens):
        self.tokens = tokens

    @abstractmethod
    def remove_stopwords(self):
        pass
