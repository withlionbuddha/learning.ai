from abc import ABC, abstractmethod
import torch

class Normalization(ABC):
    def __init__(self, encoded_data):
        self.encoded_data = encoded_data

    @abstractmethod
    def normalize(self, max_length):
        pass
