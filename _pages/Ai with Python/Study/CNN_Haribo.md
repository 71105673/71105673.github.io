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

### Accuracy, Loss
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 4.png>)
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 5.png>)

### graph
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 6.png>)
### 학습 이미지 출력
![alt text](<../../../assets/img/ARM/AI/CNN/haribo/image copy 7.png>)

## 설명
**데이터 증강:** 

데이터가 부족하거나 다양성이 적을 때, 증강을 통해 모델의 일반화 성능 향상을 기대할 수 있습니다.

**전이 학습 모델 구조:**


**콜백 설정 및 학습**
