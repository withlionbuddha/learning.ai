#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


import numpy as np
from sklearn import datasets
from IrisFlower import IrisFlower


# In[ ]:


class IrisFlowerFactory:
    def __init__(self):
        self.iris_flowers = []
        self.load_data()

    def load_data(self):
        # Iris 데이터셋 로드
        iris = datasets.load_iris()
        X = iris.data
        y_true = iris.target
        varieties = iris.target_names

        # 품종별로 IrisFlower 객체 생성
        for i, variety in enumerate(varieties):
            petal_lengths = X[y_true == i, 2]  # 꽃잎 길이
            petal_widths = X[y_true == i, 3]   # 꽃잎 넓이

            petal_length_mean = np.mean(petal_lengths)
            petal_length_std = np.std(petal_lengths)
            petal_width_mean = np.mean(petal_widths)
            petal_width_std = np.std(petal_widths)

            iris_flower = IrisFlower(
                variety=variety,
                petal_length_mean=petal_length_mean,
                petal_length_std=petal_length_std,
                petal_width_mean=petal_width_mean,
                petal_width_std=petal_width_std
            )

            self.iris_flowers.append(iris_flower)

    def get_iris_flowers(self):
        return self.iris_flowers

