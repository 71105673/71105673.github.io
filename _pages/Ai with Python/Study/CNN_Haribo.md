---
title: "Day-7 CNN_Haribo" 
date: "2025-07-01"
thumbnail: "../../../assets/img/ARM/AI/CNN/image copy 45.png"
---

# Haribo Mix Gummy

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

### graph
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 6.png>)
### 학습 이미지 출력
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 7.png>)




## DropOut(0.3), rotation_range = 45

### Accuracy, Loss (0.8763, 0.3486)
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 16.png>)
### graph
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 17.png>)
### 학습 이미지 출력
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 18.png>)


## DropOut(0.4), rotation_range = 90

### Accuracy, Loss (0.8866, 0.2899)
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 19.png>)
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 20.png>)
### graph
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



### 고찰

이번 프로젝트에서는 전체적인 데이터셋의 양이 부족해 다양한 각도나 형태에 대한 충분한 학습이 어려웠습니다. 

특히 객체의 위치나 방향이 달라질 경우 인식 정확도가 다소 낮게 나타났습니다.

하지만 데이터 증강 기법을 적용함으로써 이러한 한계를 어느 정도 보완할 수 있었고, 결과적으로 제한된 데이터셋에서도 비교적 높은 정확도로 지정한 객체를 분류할 수 있었습니다. 

이는 데이터 증강이 모델의 일반화 성능을 향상시키는 데 효과적이라는 점을 보여줍니다.