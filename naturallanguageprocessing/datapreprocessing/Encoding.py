from abc import ABC, abstractmethod

class Encoding(ABC):
    def __init__(self, tokens):
        self.tokens = tokens

    @abstractmethod
    def encode(self):
        pass
