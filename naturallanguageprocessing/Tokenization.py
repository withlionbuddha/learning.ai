from abc import ABC, abstractmethod

class Tokenization(ABC):
    def __init__(self, text):
        self.text = text

    @abstractmethod
    def tokenize(self):
        pass
