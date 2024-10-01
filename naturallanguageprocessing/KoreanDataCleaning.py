#!/usr/bin/env python
# coding: utf-8

import re
from DataCleaning import DataCleaning

class KoreanDataCleaning(DataCleaning):
    def __init__(self, text):
        super().__init__(text)
        
    def remove_special_characters(self):
        # 한국어 특수 문자 및 숫자 제거
        cleaned_text = re.sub(r"[^가-힣\s]", "", self.text)
        return cleaned_text

    def remove_html_tags(self):
        # HTML 태그 제거
        cleaned_text = re.sub(r"<[^>]+>", "", self.text)
        return cleaned_text

    def to_lowercase(self):
        # 한국어는 대소문자 구분 없음, 그대로 반환
        return self.text

    def get_text(self):
        return self.text



