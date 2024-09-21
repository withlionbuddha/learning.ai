#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/withlionbuddha/learning.ai/blob/ground/PseudoLabeling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#  😄 의사 레이블(Pseudo-label)
# 
# ---
# 의사 레이블링은 일반적으로 레이블이 일부만 주어졌을 때, 나머지 데이터를 예측하여 성능을 개선하는데 사용됩니다
# 
# 

# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


# In[ ]:


import os
import shutil
# MNIST 데이터셋이 저장된 경로
cache_dir = os.path.expanduser('~/.keras/datasets/mnist.npz')

# 캐시 파일 삭제
if os.path.exists(cache_dir):
    os.remove(cache_dir)
    print("Cached MNIST data deleted.")


# 📚 데이터셋 불러오기
# 
# ---
# 
# * mnist은 손으로 쓴 숫자(0-9) 이미지를 모아둔 데이터셋
# * train_input은 입력용 훈련이미지데이터
# * train_label은 train_input의 정답 데이터
# * test_input은 입력용 시험이미지데이터
# * test_label은 test_input의 정답 데이터
# * train_input, test_input은 0~255 사이의 정수 값(픽셀 값)으로 이루어진 배열

# In[ ]:


# MNIST 데이터셋 불러오기
(train_input, train_label), (test_input, test_label) = mnist.load_data()


# In[ ]:


print(f"train_input shape: {train_input.shape}")
print(f"axis=0 (batch of images): {train_input.shape[0]}")
print(f"axis=1 (height of each image): {train_input.shape[1]}")
print(f"axis=2 (width of each image): {train_input.shape[2]}")
print(f"train_label shape: {train_label.shape}")

print(f"-----------------------------------")
print(f"test_input shape: {test_input.shape}")
print(f"test_label shape: {test_label.shape}")


# 📚 데이터 전처리
# 
# 
# 
# ---
# 
# 
# 
# *  이미지 데이터 형태의 일관성을 위해서 channel demension을 추가한다.
# *  입력값(train_expand_input, test_input)은 astype("float32")를 사용하여 데이터를 32비트 부동 소수점 형식으로 변환합니다. 이는 신경망 모델이 실수 값을 처리하는 데 더 적합하기 때문입니다.
# * 입력(train_expand_input, test_input) 데이터의 각 이미지의 픽셀 값(0~255)을 255.0으로 나눠서 0과 1 사이의 값으로 스케일링합니다.
# 
# 
# 

# In[ ]:


# 데이터 전처리
train_expand_input = np.expand_dims(train_input, axis=-1)
test_expand_input = np.expand_dims(test_input, axis=-1)
print(f"train_expand_input shape: {train_expand_input.shape}")

train_expand_input = train_expand_input.astype("float32") / 255.0
test_expand_input = test_expand_input.astype("float32") / 255.0


# 📚 데이터 타입 확인
# 
# 
# 
# ---
# 
# 
# * numpy.ndarray 는 NumPy 라이브러리의 다차원 배열 객체로, 효율적인 수치 연산

# In[ ]:


type(train_expand_input)


# 📚 데이터 분할 , 데이터 슬라이싱
# 
# 
# 
# ---
# 
# 
# *   train_expand_input[:num_labeled]은 입력용 훈련데이터(train_expand_input)의 6000개 images 중에서 1000개의 images를 분할하여 input_labeled에 할당합니다.
# 
# 

# In[ ]:


# 레이블이 있는 데이터를 일부만 사용하고 나머지는 의사 레이블로 처리
# 여기서는 예시로 train_label의 앞부분만 사용
num_labeled = 1000
train_input_labeled = train_expand_input[:num_labeled]
train_label_labeled = train_label[:num_labeled]


# input_labeled의 형태는

# In[ ]:


print(f"train_input_labeled shape: {train_input_labeled.shape}")
print(f"train_label_labeled shape: {train_label_labeled.shape}")


# In[ ]:


test_input_unlabeled = test_expand_input[num_labeled:]
test_label_unlabeled = test_label[num_labeled:]  # test_label_unlabed는 실제 학습에 사용되지 않음

print(f"test_input_unlabeled shape: {test_input_unlabeled.shape}")
print(f"test_label_unlabeled shape: {test_label_unlabeled.shape}")


# In[ ]:


# 간단한 모델 구성
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# 먼저 라벨이 있는 데이터로 학습
model.fit(train_input_labeled , train_label_labeled , epochs=5, batch_size=32, validation_split=0.2)


# 📚 의사 레이블 생성
# 
# 의사 레이블은 예측된 결과로 새로 생성된 분류의 인덱스 입니다.
# 
# ---
# 
# *   predict()함수에서 입력용 시험데이터(test_input_unlabeled)에 대해서  각 숫자(0~9) 별로 예측된 확율값을 결과값으로 반환합니다.
# 
# *   np.argmax() 함수에서 예측된 확율결과값 중에서 가장 높은 확율값의 인덱스를 반환합니다.
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


predicted = model.predict(test_input_unlabeled)


# In[ ]:


print(f"predicted.shape : {predicted.shape}" )
print(f"len(predicted) : {len(predicted)}")
print(f"predicted[0] : {predicted[0]}")
print(f"predicted[1] : {predicted[1]}")
print(f"predicted[2] : {predicted[2]}")


# In[ ]:


# 의사 레이블 생성: 레이블이 없는 데이터를 모델을 통해 예측
pseudo_labels = np.argmax(predicted, axis=1)
pseudo_labels


# In[ ]:


len(pseudo_labels)


# In[ ]:


# 라벨이 없는 데이터에 대해 예측된 의사 레이블로 다시 학습
x_combined = np.concatenate([train_input_labeled, test_input_unlabeled ])
y_combined = np.concatenate([train_label_labeled, pseudo_labels])


# In[ ]:


model.fit(x_combined, y_combined, epochs=5, batch_size=32, validation_split=0.2)

# 테스트 데이터로 성능 확인
test_loss, test_acc = model.evaluate(train_input_labeled, train_label_labeled)
print(f"Test accuracy: {test_acc:.4f}")

