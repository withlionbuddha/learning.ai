#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dataclasses import dataclass  
from Flower import Flower


# In[2]:


@dataclass
class IrisFlower(Flower):
    def __init__(self, variety, petal_length_mean, petal_length_std, petal_width_mean, petal_width_std):
        super().__init__(
            variety=variety,
            scientific_name="Iris",      # 학명
            korean_name="붓꽃",           # 한글명
            english_name="Iris",         # 영문명
            chinese_name="鳶尾花",        # 한문명
            color="Various",             # 색상
            petal_count=6,               # 꽃잎 수
            petal_length_mean=petal_length_mean,
            petal_length_std=petal_length_std,
            petal_width_mean=petal_width_mean,
            petal_width_std=petal_width_std,
            flower_shape="Radial symmetry"  # 꽃의 모양 특징
        )


# In[ ]:




