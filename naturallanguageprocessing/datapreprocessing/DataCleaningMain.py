#!/usr/bin/env python
# coding: utf-8


from KoreanDataCleaning import KoreanDataCleaning
from EnglishDataCleaning import EnglishDataCleaning


def DataCleaningMain():
    # 예제 텍스트
    korean_text = "이것은 한국어 전처리 예제 문장입니다!"
    english_text = "This is an example sentence for preprocessing!"

    # 한국어 데이터 전처리
    korean_cleaner = KoreanDataCleaning(korean_text)
    cleaned_korean_text = korean_cleaner.remove_special_characters()
    cleaned_korean_text = korean_cleaner.remove_html_tags()
    cleaned_korean_text = korean_cleaner.to_lowercase()

    print("한국어 전처리 결과:", cleaned_korean_text)

    # 영어 데이터 전처리
    english_cleaner = EnglishDataCleaning(english_text)
    cleaned_english_text = english_cleaner.remove_special_characters()
    cleaned_english_text = english_cleaner.remove_html_tags()
    cleaned_english_text = english_cleaner.to_lowercase()

    print("영어 전처리 결과:", cleaned_english_text)

if __name__ == "__main__":
    DataCleaningMain()


# In[ ]:





# In[ ]:





# In[ ]:




