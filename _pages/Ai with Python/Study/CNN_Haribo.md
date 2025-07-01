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

## Code

```python
from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 데이터 경로 (구글 드라이브 마운트된 경로 기준)
dataset_path = '/content/drive/MyDrive/haribo_dataset'
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# 클래스 이름
class_names = ['bear', 'cola', 'egg', 'heart']

# ✅ 데이터 증강 설정 (조절된 값)
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

# 학습용 데이터 생성기
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(96, 96),  # MobileNetV2에 맞게 조정
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# 검증용 데이터 생성기
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# ✅ MobileNetV2 기반 전이학습 모델 구성
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # 처음엔 가중치 고정

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 클래스 수: 4
])

model.compile(optimizer=RMSprop(learning_rate=1e-4),
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

# ✅ 결과 시각화
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

# ✅ 증강된 학습 이미지 예시 출력
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

### Accuracy, Loss
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 4.png>)
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 4.png>)

### graph
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 6.png>)

### 학습 이미지 출력
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 7.png>)


## 설명
**데이터 증강:** 

데이터가 부족하거나 다양성이 적을 때, 증강을 통해 모델의 일반화 성능 향상을 기대할 수 있습니다.

| 파라미터                     | 기능 설명                             |
| ------------------------ | --------------------------------- |
| `rescale=1./255`         | 픽셀 값을 0~255 -> 0~1 사이로 정규화       |
| `validation_split=0.2`   | 전체 데이터의 20%를 검증(validation)용으로 사용 |
| `rotation_range=10`      | 최대 ±10도 범위로 이미지 회전                |
| `width_shift_range=0.1`  | 가로 방향으로 최대 10% 이미지 이동             |
| `height_shift_range=0.1` | 세로 방향으로 최대 10% 이미지 이동             |
| `shear_range=0.1`        | 이미지를 비스듬하게 자르는 전단(shear) 변형       |
| `zoom_range=0.1`         | 최대 ±10% 확대/축소                     |
| `horizontal_flip=True`   | 좌우 반전 허용                          |
| `fill_mode='nearest'`    | 변형 후 생긴 빈 공간을 인접 픽셀로 채움           |

**전이 학습 모델 구조:**

  - MobileNetV2: 경량 CNN 구조. 이미지넷에서 사전 학습된 가중치를 활용.

    - include_top=False: 최상위 fully-connected 분류 레이어 제외

    - trainable=False: 처음에는 base 모델의 가중치를 고정(freez)


  - GlobalAveragePooling2D: 특징 맵을 평균 풀링해서 flatten

    - Dense(128, relu): 중간 fully-connected layer

    - Dropout(0.5): 과적합 방지를 위한 정규화

    - Dense(4, softmax): 클래스 수 4개 (bear, cola, egg, heart)

**콜백 설정 및 학습**
- EarlyStopping: 검증 손실(val_loss)이 5 epoch 동안 개선되지 않으면 조기 종료

- ModelCheckpoint: 검증 성능이 최고일 때 모델 저장 (best_model.h5)
