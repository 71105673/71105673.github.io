---
title: "Day-7 Haribo" 
date: "2025-07-01"
thumbnail: "../../../assets/img/ARM/AI/CNN/image copy 45.png"
---

# Haribo Mine Project

## 객체 분류
**Heart**
![alt text](../../../assets/img/ARM/AI/CNN/haribo/image.png)

**Bear**
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 3.png>)

**Cola**
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 2.png>)

**Egg**
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy.png>)

**Ring**

![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 8.png>)

## Code

```python
from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 데이터 경로 (구글 드라이브 마운트된 경로 기준)
dataset_path = '/content/drive/MyDrive/haribo_dataset'

# Best Model 만들기
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# ✅ 클래스 이름 (ring 추가)
class_names = ['bear', 'cola', 'egg', 'heart', 'ring']

# ✅ 데이터 증강 설정
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ✅ 학습용 데이터 생성기
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# ✅ 검증용 데이터 생성기
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# ✅ MobileNetV2 기반 전이학습 모델
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')  # 클래스 수 자동 반영
])

# ✅ 옵티마이저 변경 (Adam)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ 콜백 설정
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# ✅ 학습 실행
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stop, checkpoint],
    verbose=2
)

# ✅ 정확도 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# ✅ 증강된 이미지 샘플 시각화
x_batch, y_batch = next(train_generator)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(x_batch[i])
    label_idx = np.argmax(y_batch[i])
    plt.xlabel(class_names[label_idx])
plt.tight_layout()
plt.show()
```




## 결과

## DropOut(0.5), rotation_range = 10

### Accuracy, Loss (0.8351, 0.4658)
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 4.png>)
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 5.png>)
### graph 1
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 6.png>)
### 학습 이미지 출력
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 7.png>)




## DropOut(0.3), rotation_range = 45

### Accuracy, Loss (0.8763, 0.3486)
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 16.png>)
### graph 2
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 17.png>)
### 학습 이미지 출력
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 18.png>)


## DropOut(0.4), rotation_range = 90

### Accuracy, Loss (0.8866, 0.2899)
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 19.png>)
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 20.png>)
### graph 3
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 21.png>)
### 학습 이미지 출력
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 22.png>)


# 설명

## 데이터 증강 

훈련 이미지의 다양성을 인위적으로 늘려서 **과적합(overfitting)**을 방지하고 모델이 더 일반화(generalization)되도록 돕기 위해 사용합니다.

📌 주요 파라미터 설명:

    rescale=1./255
    → 픽셀 값(0255)을 **01로 정규화**해줌. 신경망 학습 안정성을 높이기 위해 필요합니다.

    validation_split=0.2
    → 전체 데이터 중 20%를 검증용, 나머지를 학습용으로 자동 분리합니다.

    rotation_range=10
    → 이미지를 최대 ±10도 회전. 회전에 강한 모델을 만듭니다.

    width_shift_range=0.1, height_shift_range=0.1
    → 이미지의 수평/수직 방향으로 10% 이동. 위치에 덜 민감하도록 합니다.

    shear_range=0.1
    → 비스듬히 기울이는 변형(전단 shear)을 적용합니다.

    zoom_range=0.1
    → 확대/축소를 통해 크기 변화에 적응하는 모델을 만듭니다.

    horizontal_flip=True
    → 좌우 반전. 비대칭적인 데이터도 잘 학습하게 합니다.

    fill_mode='nearest'
    → 이미지 이동/회전 시 생긴 빈 부분은 가장 가까운 픽셀 값으로 채웁니다.

## 전이 학습 모델 구조

🎯 MobileNetV2란?

    ImageNet 데이터셋으로 미리 학습된 경량 CNN 모델

    모바일이나 임베디드 환경에서도 성능과 속도 균형이 좋음

📌 주요 설정:

    include_top=False:
    → MobileNetV2의 마지막 fully connected 분류 층은 제외하고, **특징 추출기(feature extractor)**로만 사용합니다.

    weights='imagenet':
    → ImageNet에서 학습된 가중치를 가져와 사용합니다.

    base_model.trainable = False:
    → 기존에 학습된 가중치를 고정(freeze). 새로운 데이터셋에서 다시 학습하지 않음.

GlobalAveragePooling2D:

    평균을 이용하여 전체 특징 맵을 벡터로 바꿔주는 층. Flatten보다 과적합에 덜 민감함.

Dropout:

    일부 뉴런을 랜덤하게 제거하여 일반화 성능 향상.

Softmax 출력층:

    다중 클래스 분류 문제이므로 softmax 사용.

## 콜백 설정 및 학습

🎯 콜백(CallBacks)이란?

모델 학습 도중 특정 조건을 만족할 때 자동으로 동작하는 기능 (예: 학습 중단, 모델 저장 등)

📌 EarlyStopping

    monitor='val_loss':
    검증 손실(val_loss)을 기준으로 모니터링

    patience=5:
    검증 손실이 5번 연속 개선되지 않으면 학습 중단

    restore_best_weights=True:
    가장 성능이 좋았던 시점의 가중치로 복원

📌 ModelCheckpoint

    save_best_only=True:
    검증 성능이 좋아질 때만 모델을 저장 → best_model.h5로 저장됨

















# 카메라 확인

```python
import cv2
import numpy as np
import tensorflow as tf
import json

# 모델과 클래스 이름 로드
model = tf.keras.models.load_model('best_model.h5')

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

def preprocess(frame):
    img = cv2.resize(frame, (96, 96))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("젤리 분류 시작! (Q 키를 누르면 종료)")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_img = preprocess(frame)
    pred = model.predict(input_img)
    label = class_names[np.argmax(pred)]

    # 예측 결과 화면에 출력
    cv2.putText(frame, f'Prediction: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Haribo Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 결과

![text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 9.png>) ![text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 10.png>) ![text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 11.png>) ![text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 12.png>) ![text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 13.png>) ![text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 14.png>) ![text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 15.png>) ![text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 46.png>)

학습한 Best_model.h5를 기준으로 확인한결과 객체를 잘 인식하는 모습을 확인할 수 있다. 

추가로 진행한 DropOut(0.5 -> 0.3), rotation_range = 10 -> 45 의 경우 Best_model_2.h5로 진행하였지만, DropOut이 과도하게 적용 및 증강 강도가 증가함에 따라 튀는 경향이 증가했습니다. 

따라서 DropOut = 0.4, rotation_range = 90으로 설정한 경우
안정적인 학습 곡선과 함께 좋은 일반화 성능을 가질 수 있었고, 가장 좋은 인식률을 보였습니다.

## 결론 및 고찰

✅ 1. 모델별 성능 요약

| Graph | DropOut | Rotation Range | Accuracy   | Loss       | 특징                                    |
| ----- | ------- | -------------- | ---------- | ---------- | ------------------------------------- |
| 1     | 0.5     | 10             | 0.8351     | 0.4658     | 훈련 정확도는 높지만 검증 정확도와 손실 변동 큼 → 과적합 가능성 |
| 2     | 0.3     | 45             | 0.8763     | 0.3486     | 전반적으로 안정된 성능. Graph 1보다 개선            |
| 3     | 0.4     | 90             | **0.8866** | **0.2899** | 가장 높은 정확도와 가장 낮은 손실. 학습 안정성 및 일반화 뛰어남 |

✅ 2. 최우수 모델: Graph 3

    DropOut: 0.4

    Rotation Range: 90

    최종 정확도: 0.8866

    최종 손실: 0.2899

    특징: 훈련/검증 정확도 모두 안정적으로 수렴하며 과적합이 가장 적은 모델로 평가됨.

✅ 3. 결론

    DropOut 0.4와 rotation_range 90 조합은 모델 성능 향상에 효과적이었음.

    Graph 3은 높은 정확도, 낮은 손실, 안정된 학습 곡선을 보여주며 최적의 조합으로 판단됨.

    적절한 하이퍼파라미터 튜닝이 모델 성능에 직접적인 영향을 미친다는 것을 확인함.

### 📉 데이터 부족과 일반화 성능 저하

- **데이터 양 부족**  
  - 학습에 사용 가능한 데이터의 양이 충분하지 않아 모델이 다양한 경우를 일반화하기 어려웠습니다.

- **단순 증강의 한계**  
  - 회전, 이동 등의 기본적인 데이터 증강 기법만으로는 형태의 다양성을 충분히 확보하기 어려웠습니다.

- **형태 기반 분류 문제의 특성**  
  - 입력 데이터의 **각도, 배경 변화** 등에 매우 민감하여, 실제 테스트 시 오류가 자주 발생했습니다.

- **과적합 발생**  
  - 훈련 정확도는 높았지만, 검증 및 테스트에서는 성능이 크게 저하되어 **일반화 능력 부족**이 드러났습니다.

---

### 🔄 전이 학습의 필요성

- **MobileNetV2 활용**  
  - 경량화된 구조의 MobileNetV2를 사용하여 **제한된 데이터 환경**에서도 안정적인 학습이 가능했습니다.

- **높은 연산 효율과 빠른 처리 속도**  
  - 연산량이 적어 여러 실험을 **빠르게 반복**할 수 있었으며, 실험 효율성이 향상되었습니다.

- **작은 모델에서의 우수한 성능**  
  - 모델 크기는 작지만, 일정 수준 이상의 **형태 분류 정확도**를 안정적으로 유지할 수 있었습니다.


### 🧠 고찰

#### DropOut과 데이터 증강의 시너지

- 초기에는 회전 등 **단순한 증강 기법만으로는 데이터 다양성 확보가 어려워 일반화 성능이 낮았습니다.**
- 이후, **DropOut(0.4)**을 적용하고, **rotation_range=90**과 같은 **증강 기법**을 도입하면서 모델은 더 다양한 형태에 대응할 수 있게 되었고, **과적합 억제와 일반화 성능 향상**에 효과적이었습니다.

#### 하이퍼파라미터 튜닝의 중요성

- DropOut 비율과 증강 범위 등 **하이퍼파라미터를 적절히 조정**하는 것이 모델 성능 향상에 **결정적인 역할**을 했습니다.
  
#### 과적합 방지 효과

- 훈련과 검증 정확도 간의 차이가 줄어들며, **안정적인 학습이 가능해졌고**, 실제 테스트에서도 **높은 정확도와 재현성**을 유지했습니다.


---

## 정보! MLP보다 CNN이 좋은 이유

#### 1. 지역 특성(Local Features) 추출
- **CNN은 이미지의 지역적인 패턴(예: 선, 모서리)**을 **커널(필터)**을 통해 추출합니다.  
- 반면 **MLP는 입력을 전부 한 줄로 펼쳐서(flatten)** 처리하기 때문에 공간 구조나 위치 정보를 활용하지 못합니다.

> 예: 고양이 사진에서 귀나 눈 같은 부분 특징을 CNN은 잘 잡지만, MLP는 픽셀의 순서가 바뀌면 큰 혼란을 겪습니다.

---

#### 2. 파라미터 수가 훨씬 적음 (효율성)
- MLP는 모든 입력 노드가 모든 은닉층 노드와 연결되기 때문에 **파라미터 수가 매우 큽니다.**
- CNN은 **커널(필터)을 공유**하면서 **국소 영역만 학습**하기 때문에 학습해야 할 파라미터 수가 적고, 계산량도 훨씬 작습니다.

> 예: 256x256 이미지를 MLP로 처리하면 수백만 개의 weight가 필요하지만, CNN은 수천~수만 개로 충분합니다.

---

#### 3. 위치 변화에 강함 (Translation Invariance)
- CNN은 **pooling 등의 구조**로 인해 **이미지의 위치가 조금 달라져도 잘 인식**합니다.
- MLP는 입력이 조금만 바뀌어도 **완전히 다른 결과를 낼 수 있어**, 위치 변화에 매우 민감합니다.

---

#### 4. 계층적 특징 추출 (Hierarchical Feature Learning)
- CNN은 **저수준(모서리, 점) → 중간수준(도형, 패턴) → 고수준(사물, 객체)** 으로 **계층적인 특징 추출**이 가능합니다.
- MLP는 이런 **구조적 특징 학습이 어렵고**, 복잡한 패턴을 잘 파악하지 못합니다.
