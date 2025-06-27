---
title: "Day-5 ANN" 
date: "2025-06-27"
thumbnail: "../../../assets/img/ARM/AI/image copy 32.png"
---
# 선형 모델 학습

## 선형 모델 학습 과정
![alt text](<../../../assets/img/ARM/AI/image copy 37.png>)

## 벡터화 
- 선형 모델에서는 입력 데이터는 벡터 형태로 정리
- 2차원 또는 3차원 이미지 데이터를 1차원 벡터로 변환
- 선형 모델에서는 입력이 반드시 1차원 벡터, 따라서 벡터화 필수이다.
- 다음은 4X4 픽셀 이미지 예시이다
![alt text](<../../../assets/img/ARM/AI/image copy 38.png>)

## 벡터화 코드

```python
import numpy as np

# 0~255 사이의 임의의 정수로 구성된 4x4 행렬 생성
a = np.random.randint(0, 255, (4, 4))
print("원본 4x4 행렬:")
print(a)

# flatten을 사용해 1차원 배열(벡터)로 변환
b = a.flatten()
print("\nFlatten된 1차원 배열:")
print(b)

# reshape을 사용해 행렬 크기를 변경
# -1은 자동 계산되며, 이 경우 총 원소 수가 16이므로 reshape(-1)은 (16,)과 동일
# 예: (2, 8)로 바꾸고 싶다면 reshape(2, -1) 또는 reshape(2, 8) 모두 가능
c = a.reshape(-1)
print("\nReshape(-1) 결과:")
print(c)
```
## 선형 분류기 - Score 함수

**score = W·x + b**

- x: 입력 벡터 (flatten된 이미지)

- W: 가중치 행렬 (클래스 수 × 입력 특성 수)

- b: 바이어스 벡터

- score: 각 클래스에 대한 점수 (score vector)

![alt text](<../../../assets/img/ARM/AI/image copy 39.png>)

**병렬처리**

+ X: m개의 입력 샘플 (m, n), W: 가중치 (k, n)

+ 결과 S: score (m, k)

+ S = np.dot(X, W.T) + b

![alt text](<../../../assets/img/ARM/AI/image copy 40.png>)

## Sofemax 분류기
Softmax는 각 클래스의 점수를 확률로 변환합니다:
![alt text](<../../../assets/img/ARM/AI/image copy 45.png>)

s_j: 클래스 j의 점수 (score)

전체 클래스의 score를 softmax에 통과시켜 확률 분포로 만듭니다.

![alt text](<../../../assets/img/ARM/AI/image copy 41.png>)

**진행 과정**
  
![alt text](<../../../assets/img/ARM/AI/image copy 42.png>)

**ross Entropy Loss 과정**

Softmax의 출력 결과와 실제 정답 간의 차이를 측정하는 손실 함수:

![alt text](<../../../assets/img/ARM/AI/image copy 47.png>)

y_true 위치의 softmax 확률에 -log를 취한 값

정답 클래스의 확률이 높을수록 loss가 낮아짐

![alt text](<../../../assets/img/ARM/AI/image copy 43.png>)

**최적화: SGD**

학습은 경사 하강법을 기반으로 최적화됩니다:

    Full Gradient Descent는 전체 데이터를 사용하지만 연산량이 많음

    대신 **Stochastic Gradient Descent (SGD)**는 일부 배치만 사용:

작은 배치(mini-batch) 단위로 파라미터를 조금씩 업데이트하면서 전체적으로 손실을 줄여나가는 방식

![alt text](<../../../assets/img/ARM/AI/image copy 43.png>)


## 🔍 전체 학습 흐름 요약

    이미지 → 벡터화

    벡터 → Score 계산 (Wx + b)

    Score → Softmax → 확률 분포

    예측 확률과 실제 라벨로 Cross Entropy Loss 계산

    Loss에 따라 SGD로 파라미터 업데이트



# Mnist 실습
```python
import numpy as np
import pandas as pd

from tensorflow.keras.datasets.mnist import load_data
(train_x, train_y), (test_x, test_y) = load_data()

train_x.shape, train_y.shape, # Train 데이터 크기 확인
test_x.shape, test_y.shape # Test 데이터 크기 확인
```
> 결과 : ((10000, 28, 28), (10000,))
```python
# 이미지 확인하기
from PIL import Image
img=train_x[0]

import matplotlib.pyplot as plt
img1 = Image.fromarray(img, mode = 'L')
plt.imshow(img1)

train_y[0]
```
> 결과:![alt text](<../../../assets/img/ARM/AI/image copy 48.png>)
```python
# 데이터 전처리

## 입력 형태 변환: 3-> 2 차원
### 데이터를 2차원 형태로 변환: 입력 데이터가 선형 모델에서는 벡터 형태
train_x1 = train_x.reshape(60000, -1)
test_x1 = test_x.reshape(10000, -1)

### 데이터 값의 크기 조절: 0~1 사이 값으로 변환
train_x2 = train_x1 / 255
test_x2 = test_x1 / 255
```
```python
# 모델 설정

## 라이브러리 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## 모델 설정
md = Sequential()
md.add(Dense(10, activation='softmax', input_shape=(28*28,)))
md.summary()  # 모델 요약
```
> 결과:![alt text](<../../../assets/img/ARM/AI/image copy 49.png>)

```python
# 모델 학습 진행
## 모델 complile: 손실 함수, 최적화 함수, 측정 함수 설정
md.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['acc'])

## 모델 학습: 학습 횟수. batch_size, 검증용 데이터 설정
hist = md.fit(train_x2, train_y, epochs=30, batch_size=64, validation_split=0.2)
```
```
결과:
Epoch 1/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.5941 - loss: 1.4901 - val_acc: 0.8569 - val_loss: 0.6543
Epoch 2/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - acc: 0.8506 - loss: 0.6448 - val_acc: 0.8757 - val_loss: 0.5063
Epoch 3/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.8672 - loss: 0.5302 - val_acc: 0.8859 - val_loss: 0.4481
Epoch 4/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - acc: 0.8812 - loss: 0.4678 - val_acc: 0.8905 - val_loss: 0.4164
Epoch 5/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - acc: 0.8831 - loss: 0.4416 - val_acc: 0.8957 - val_loss: 0.3948
Epoch 6/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.8844 - loss: 0.4267 - val_acc: 0.8987 - val_loss: 0.3801
Epoch 7/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - acc: 0.8912 - loss: 0.4030 - val_acc: 0.9002 - val_loss: 0.3687
Epoch 8/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.8933 - loss: 0.3907 - val_acc: 0.9028 - val_loss: 0.3596
Epoch 9/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - acc: 0.8933 - loss: 0.3876 - val_acc: 0.9030 - val_loss: 0.3526
Epoch 10/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - acc: 0.8954 - loss: 0.3798 - val_acc: 0.9061 - val_loss: 0.3466
Epoch 11/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.8990 - loss: 0.3698 - val_acc: 0.9076 - val_loss: 0.3411
Epoch 12/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9017 - loss: 0.3588 - val_acc: 0.9087 - val_loss: 0.3369
Epoch 13/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.8977 - loss: 0.3606 - val_acc: 0.9088 - val_loss: 0.3329
Epoch 14/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - acc: 0.9024 - loss: 0.3509 - val_acc: 0.9093 - val_loss: 0.3294
Epoch 15/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9030 - loss: 0.3540 - val_acc: 0.9097 - val_loss: 0.3261
Epoch 16/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9023 - loss: 0.3441 - val_acc: 0.9097 - val_loss: 0.3237
Epoch 17/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9039 - loss: 0.3392 - val_acc: 0.9120 - val_loss: 0.3209
Epoch 18/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9045 - loss: 0.3394 - val_acc: 0.9114 - val_loss: 0.3185
Epoch 19/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - acc: 0.9081 - loss: 0.3316 - val_acc: 0.9131 - val_loss: 0.3165
Epoch 20/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9072 - loss: 0.3361 - val_acc: 0.9129 - val_loss: 0.3144
Epoch 21/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9093 - loss: 0.3287 - val_acc: 0.9137 - val_loss: 0.3124
Epoch 22/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 4s 3ms/step - acc: 0.9061 - loss: 0.3333 - val_acc: 0.9141 - val_loss: 0.3110
Epoch 23/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9098 - loss: 0.3243 - val_acc: 0.9144 - val_loss: 0.3093
Epoch 24/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9109 - loss: 0.3231 - val_acc: 0.9152 - val_loss: 0.3078
Epoch 25/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9099 - loss: 0.3222 - val_acc: 0.9159 - val_loss: 0.3065
Epoch 26/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - acc: 0.9094 - loss: 0.3223 - val_acc: 0.9156 - val_loss: 0.3052
Epoch 27/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9093 - loss: 0.3212 - val_acc: 0.9161 - val_loss: 0.3039
Epoch 28/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9124 - loss: 0.3136 - val_acc: 0.9164 - val_loss: 0.3027
Epoch 29/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9122 - loss: 0.3174 - val_acc: 0.9168 - val_loss: 0.3016
Epoch 30/30
750/750 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9140 - loss: 0.3116 - val_acc: 0.9178 - val_loss: 0.3005
```

```python
acc = hist.history['acc']
val_acc = hist.history['val_acc']
epoch = np.arange(1, len(acc)+1)

# 학습 결과 분석: 학습 곡선 그리기
plt.figure(figsize=(10, 8))
plt.plot(epoch, acc, 'b', label='Training accuracy')
plt.plot(epoch, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
> 결과: ![alt text](<../../../assets/img/ARM/AI/image copy 50.png>)
```python
# 테스트용 데이터 평가
md.evaluate(test_x2, test_y)

# 가중치 저장
weight = md.get_weights()
weight
```
> 결과: ![alt text](<../../../assets/img/ARM/AI/image copy 51.png>)
```python
# Model Loss 시각화
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```
> 결과:![alt text](<../../../assets/img/ARM/AI/image copy 52.png>)

















# ANN 모델 실습

## 접근 
>mkdir F_MNIST
>
>cd F_MNIST
>
>python3 -m venv .fmnist
>
>source .fmnist/bin/activate
>
>pip install tensorflow matplotlib PyQt5 scikit-learn
>
>export QT_QPA_PLATFORM=wayland -> 터미널 오픈시 실행
>
>python fs_mnist.py

## fs_mnist.py

```python
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# dataset load
fashion_mnist = keras.datasets.fashion_mnist

# spilt data (train / test)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
               
matplotlib.use('Qt5Agg')
NUM=20
plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=1)
for idx in range(NUM):
    sp = plt.subplot(5,5,idx+1)
    plt.imshow(train_images[idx])
    plt.title(f'{class_names[train_labels[idx]]}')
plt.show()

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 간단한 이미지 전처리 (for ANN)
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,8))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential ([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)

predictions = model.predict(test_images)

predictions[0]

np.argmax(predictions[0])

test_labels[0]

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

from sklearn.metrics import accuracy_score
print('accuracy score : ', accuracy_score(tf.math.argmax(predictions, -1), test_labels))
```