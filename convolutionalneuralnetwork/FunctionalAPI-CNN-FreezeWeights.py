#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/withlionbuddha/learning.ai/blob/ground/FunctionalAPI-CNN-FreezeWeights.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 📚
# 가중치 고정(Freeze Weights) <br>
# 기존 모델의 특정 레이어(주로 하위 레이어)의 가중치를 고정하고, 새로운 데이터셋에 대해 상위 레이어만 학습하는 방식입니다. 하위 레이어는 일반적인 특징(예: 엣지, 질감 등)을 추출하고, 상위 레이어는 더 구체적인 패턴을 학습합니다.

# In[ ]:





# 📚
# 
# 예시: 이미지 분류 모델
# 
# 기존 모델: ImageNet 데이터셋에서 사전 학습된 ResNet 같은 대형 CNN 모델.
# 새로운 데이터셋: 고양이와 개 이미지를 분류하는 소규모 데이터셋.
# 
# 절차:
# 
# 모델의 **하위 레이어(초기 레이어)**에서 이미지의 기본적인 특징(엣지, 텍스처 등)을 추출하는데 이미 충분히 학습되었으므로, 이를 고정하고 가중치를 업데이트하지 않음.
# 상위 레이어에서 새로운 데이터셋에 맞춘 학습을 진행해, 고양이와 개의 고유한 특징을 학습.
# 이점:
# 
# 연산 비용이 절약되며, 적은 양의 데이터로도 높은 성능을 얻을 수 있습니다.
# 모델의 일반화 능력이 유지되며, 기존에 학습된 일반적인 특징을 활용할 수 있습니다.

# In[1]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# In[ ]:





# 📚 아래의 코드 설명
# 
# *   weights='imagenet'는 transfer learnig 에서 사전 학습된 가중치를 불러오기 위한 설정
# *   include_top=False는 상단의 fully-connected layer를 제거한다는 의미
# 

# In[2]:


# ResNet50 모델을 불러오고 ImageNet에서 학습된 가중치를 사용
base_model = ResNet50(weights='imagenet', include_top=False)


# 

# In[3]:


base_model


# 📚 아래의 코드 설명
# 
# *   base_model.layers 는 base_model의 layer 속성
# 

# In[4]:


base_model.layers


# 📚 아래의 코드 설명
# 
# *   layer.trainable 은 Keras/TensorFlow에서 각 레이어가 학습 중에 가중치가 업데이트될 수 있는지 여부를 결정하는 속성
# *   trainable=True: 이 레이어는 학습 중에 가중치가 업데이트 합니다.
# 
# *   trainable=False: 이 레이어는 학습 중에 가중치가 업데이트되지 않습니다. 즉, 현재 가중치가 고정된 상태로 유지 됩니다.

# In[5]:


# 가중치를 고정 (이전 레이어의 학습을 멈춤)
for layer in base_model.layers:
    layer.trainable = False


# In[6]:


base_model.output


# In[7]:


# 새로운 레이어를 추가하여 특정 문제에 맞게 학습
pooling_output = GlobalAveragePooling2D()(base_model.output)
dense_output = Dense(1024, activation='relu')(pooling_output)
prediction_output = Dense(10, activation='softmax')(dense_output)  # 10개의 클래스를 예측하는 레이어


# 📚 아래의 코드 설명
# 
# 

# In[8]:


# 새로운 모델 구성
model = Model(inputs=base_model.input, outputs=prediction_output)

# 모델 컴파일 (고정된 레이어는 학습되지 않음)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 새로운 데이터셋으로 학습
# train_data와 train_labels는 새로 학습할 데이터셋
# model.fit(train_data, train_labels, epochs=10)

