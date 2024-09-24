#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dataclasses import dataclass


# In[2]:


@dataclass
class Flower:
    variety: str = None                # 품종명
    scientific_name: str = None        # 학명
    korean_name: str = None            # 한글명
    english_name: str = None           # 영문명
    chinese_name: str = None           # 한문명
    color: str = None                  # 색상
    petal_count: int = None            # 꽃잎 수
    petal_length_mean: float = None    # 꽃잎 길이 평균(cm)
    petal_length_std: float = None     # 꽃잎 길이 표준편차(cm)
    petal_width_mean: float = None     # 꽃잎 넓이 평균(cm)
    petal_width_std: float = None      # 꽃잎 넓이 표준편차(cm)
    flower_shape: str = None           # 꽃의 모양 특징

    def __str__(self):
        return (
            f"품종명: {self.variety}\n"
            f"학명: {self.scientific_name}\n"
            f"한글명: {self.korean_name}\n"
            f"영문명: {self.english_name}\n"
            f"한문명: {self.chinese_name}\n"
            f"색상: {self.color}\n"
            f"꽃잎 수: {self.petal_count}\n"
            f"꽃잎 길이 평균(cm): {self.petal_length_mean}\n"
            f"꽃잎 길이 표준편차(cm): {self.petal_length_std}\n"
            f"꽃잎 넓이 평균(cm): {self.petal_width_mean}\n"
            f"꽃잎 넓이 표준편차(cm): {self.petal_width_std}\n"
            f"꽃의 모양 특징: {self.flower_shape}"
        )


# In[ ]:




