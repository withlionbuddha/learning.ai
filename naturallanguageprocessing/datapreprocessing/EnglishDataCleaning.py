#!/usr/bin/env python
# coding: utf-8


import re
from DataCleaning import DataCleaning

class EnglishDataCleaning(DataCleaning):
    def remove_special_characters(self):
        # 영어 특수 문자 및 숫자 제거
        cleaned_text = re.sub(r"[^a-zA-Z\s]", "", self.text)
        return cleaned_text

    def remove_html_tags(self):
        # HTML 태그 제거
        cleaned_text = re.sub(r"<[^>]+>", "", self.text)
        return cleaned_text

    def to_lowercase(self):
        # 영어는 대소문자 통일
        return self.text.lower()

    def get_text(self):
        return self.text




