#!/usr/bin/env python
# coding: utf-8



from abc import ABC, abstractmethod

class DataCleaning(ABC):
    def __init__(self, text):
        self.text = text

    @abstractmethod
    def remove_special_characters(self):
        pass

    @abstractmethod
    def remove_html_tags(self):
        pass

    @abstractmethod
    def to_lowercase(self):
        pass
    
    @abstractmethod
    def get_text(self):
        return self.text



