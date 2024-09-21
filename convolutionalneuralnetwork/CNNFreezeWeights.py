#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/withlionbuddha/learning.ai/blob/ground/CNN%20FreezeWeights.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 📚
# 가중치 고정(Freeze Weights)
# 기존 모델의 특정 레이어(주로 하위 레이어)의 가중치를 고정하고, 새로운 데이터셋에 대해 상위 레이어만 학습하는 방식입니다. 하위 레이어는 일반적인 특징(예: 엣지, 질감 등)을 추출하고, 상위 레이어는 더 구체적인 패턴을 학습합니다.

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

# In[ ]:


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

# In[ ]:


# ResNet50 모델을 불러오고 ImageNet에서 학습된 가중치를 사용
base_model = ResNet50(weights='imagenet', include_top=False)


# 

# In[ ]:


base_model


# 📚 아래의 코드 설명
# 
# *   base_model.layers 는 base_model의 layer 속성
# 

# In[ ]:


base_model.layers


# 📚 아래의 코드 설명
# 
# *   layer.trainable 은 Keras/TensorFlow에서 각 레이어가 학습 중에 가중치가 업데이트될 수 있는지 여부를 결정하는 속성
# *   trainable=True: 이 레이어는 학습 중에 가중치가 업데이트 합니다.
# 
# *   trainable=False: 이 레이어는 학습 중에 가중치가 업데이트되지 않습니다. 즉, 현재 가중치가 고정된 상태로 유지 됩니다.

# In[ ]:


# 가중치를 고정 (이전 레이어의 학습을 멈춤)
for layer in base_model.layers:
    layer.trainable = False


# 
# 📚 아래의 코드 설명
# 
# *   Chaining은 각 레이어의 출력은 다음 레이어의 입력으로 사용되며, 이를 체이닝 방식으로 연결해 나가는 것
# 
# 
# 
# *   Chaining 코딩 방식은 변수 tensor_values에 결과값을 연속적으로 대입하는방식
# *    tensor_values 에는 base_model.output 의 메모리 주소값이 저장됩니다.tensor_values는 base_model.output 텐서의 메모리 주소를 참조합니다.
# *   GlobalAveragePooling2D()(tensor_values) 에서 tensor_values가 입력값으로 대입되어  결과값으로 텐서가 생성되면 tensor_values 에는 새롭게 생성된 GlobalAveragePooling2D()(tensor_values)텐서의 메모리 주소값이 저장됩니다.
# 
# *   base_model.output는 사전 학습된 모델(예: ResNet, VGG)의 최종 레이어 출력(특징 맵)
# *   GlobalAveragePooling2D()(tensor_values) 는 특징 맵 tensor_values에 평균 풀링을 적용하여 특징 맵의 차원을 줄어들게 하여 압축된 정보.
# 
# *  Dense(1024, activation='relu')(tensor_values)는 압축된 정보 tensor_values 를 입력값, 1024개의 뉴런, 입력 값이 0보다 크면 그대로 출력하고, 0보다 작으면 0을 출력하는 활성화 함수 relu로 구성된 fully connected layer.
# *   Dense(10, activation='softmax')(tensor_values) 는  Dense(1024, activation='relu')(tensor_values)의 결과인 텐서값을 입력값, 10개의 뉴런, 모든 입력 값을 0에서 1 사이의 값으로 변환하고, 그 합이 1이 되도록 활성화 함수 softmax 으로 구성된 fully connected layer.
# 
# ---
# 
# tensor_values 가 덮여쓰여지도록 만드는 chaining 코딩방식.
# 
# ---

# In[ ]:


# 새로운 레이어를 추가하여 특정 문제에 맞게 학습
tensor_values = base_model.output
tensor_values = GlobalAveragePooling2D()(tensor_values)
tensor_values = Dense(1024, activation='relu')(tensor_values)
predictions = Dense(10, activation='softmax')(tensor_values)  # 10개의 클래스를 예측하는 레이어


# 📚 아래의 코드 설명
# 
# 

# In[ ]:


# 새로운 모델 구성
model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일 (고정된 레이어는 학습되지 않음)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 새로운 데이터셋으로 학습
# train_data와 train_labels는 새로 학습할 데이터셋
# model.fit(train_data, train_labels, epochs=10)

