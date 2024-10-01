from abc import ABC, abstractmethod

class Lemmatization(ABC):
    def __init__(self, tokens):
        self.tokens = tokens

    @abstractmethod
    def apply_lemmatization(self):
        pass
