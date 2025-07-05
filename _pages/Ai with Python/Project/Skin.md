---
title: "Skin_Check" 
date: "2025-07-01"
thumbnail: "../../../assets/img/ARM/AI/skin/images.png"
---

# <Google Colab 학습 Code>

```python
# SKIN_Project.ipynb
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ✅ Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# ✅ 경로 설정
dataset_path = '/content/drive/MyDrive/SKIN/dataset_skin'  # 너가 올린 경로로 수정
model_save_path = '/content/drive/MyDrive/SKIN/skin_model.h5'  # 원하는 저장 경로

# ✅ 데이터 증강 설정 (rotation_range는 15도만 줌)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=90,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    fill_mode='nearest'
)

# ✅ 데이터 로딩
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(96, 96),  # MobileNetV2는 최소 96x96부터 가능
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# ✅ 클래스 이름 자동 추출
class_names = list(train_generator.class_indices.keys())
print("클래스 인덱스:", train_generator.class_indices)

# ✅ MobileNetV2 기반 모델 구성
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

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

# ✅ 학습 이미지 예시
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

# ✅ 모델 저장 (.h5 파일)
model.save(model_save_path)
print(f"모델이 저장되었습니다: {model_save_path}")
```

## Result
![alt text](<../../../assets/img/ARM/AI/skin/스크린샷 2025-07-04 204712.png>)

## Ubuntu Terminal Dir
```json
// skin_names.json
[
  "기저세포암",
  "보웬병",
  "표피낭종",
  "비립종",
  "정상피부",
  "화농성 육아종",
  "편평세포암",
  "사마귀"
]
```
## predict_cam.py
``` python
# predict_cam.py
import cv2
import numpy as np
import tensorflow as tf
import json
from PIL import ImageFont, ImageDraw, Image

# 클래스 이름 (한글) 불러오기
with open("skin_names.json", "r") as f:
    class_names = json.load(f)

# 모델 로드
model = tf.keras.models.load_model("skin_model.h5")

# 한글 폰트 경로 (Ubuntu에서 사용 가능)
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font = ImageFont.truetype(FONT_PATH, 32)

# 웹캠 시작
cap = cv2.VideoCapture(2)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 예측을 위한 전처리
    img = cv2.resize(frame, (96, 96))
    img_array = np.expand_dims(img / 255.0, axis=0)
    pred = model.predict(img_array)[0]
    label = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    # 원본 프레임을 Pillow 이미지로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    text = f"{label} ({confidence*100:.1f}%)"

    # 텍스트 출력
    draw.text((10, 30), text, font=font, fill=(0, 255, 0))  # 초록색

    # 다시 OpenCV 형식으로 변환하여 출력
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Skin Classifier", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

# 라즈베리파이 피부 질환 진단 및 상담 시스템

실시간 카메라 영상으로 15종의 피부 질환을 예측하고, Ollama와 Gemma3 모델을 통해 관련 정보를 제공하는 시스템입니다.

## 폴더 구조
```
skin_diagnosis_pi/
├── app.py
├── requirements.txt
├── model/
│   └── best_skin_disease_model.pth
├── font/
│   └── NanumGothic.ttf
└── README.md
```

## 라즈베리파이 설정 단계

### 1. 프로젝트 폴더 복사
- 이 `skin_diagnosis_pi` 폴더 전체를 라즈베리파이의 원하는 위치로 복사합니다. (예: `/home/pi/`)

### 2. 필수 라이브러리 설치
- 터미널을 열고 프로젝트 폴더로 이동합니다.
  ```bash
  cd /path/to/skin_diagnosis_pi
  ```
  가상환경 이동 후,
- `requirements.txt` 파일을 이용해 모든 파이썬 라이브러리를 설치합니다.
  ```bash
  pip3 install -r requirements.txt
  ```
- **중요**: 라즈베리파이에서는 `torch`와 `torchvision` 설치가 까다로울 수 있습니다. 위 명령이 실패하면, 라즈베리파이 OS 버전에 맞는 파일을 직접 찾아 설치해야 할 수 있습니다. [이곳](https://github.com/KinaIso/RaspberryPi-Pytorch-builder)과 같은 커뮤니티 빌드를 참고하세요.

### 3. Ollama 설치 및 Gemma3 모델 다운로드
- [Ollama 공식 홈페이지](https://ollama.com/download)로 이동하여 라즈베리파이용 Ollama를 설치합니다.
ollama run gemma3:1b 
- 설치가 완료되면 터미널에서 아래 명령을 실행하여 `gemma3` 모델을 다운로드합니다.
  ```bash
  ollama run gemma3:1b
  ```
- 위 명령을 실행하면 모델 다운로드 후 `>>>` 프롬프트가 나타납니다. 이 터미널은 **닫지 말고 그대로 켜두세요.** Ollama 서버 역할을 합니다.

## 실행 방법

1.  **새로운 터미널 창**을 엽니다.
2.  프로젝트 폴더로 이동합니다.
   ```bash
   cd /path/to/skin_diagnosis_pi
   ```
3.  메인 스크립트를 실행합니다.
   ```bash
   python3 app.py
   ```
4.  카메라 영상이 나타나면, 진단할 부위를 중앙에 맞추고 키보드 `c`를 눌러 진단을 시작하세요.
5.  종료하려면 `q`를 누릅니다.


## requirements.txt
```
opencv-python
torch
torchvision
numpy
Pillow
ollama
```

## app.py
```python


import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import time
import os
import ollama

# --- 설정 ---
CAPTURE_INTERVAL = 1  # 캡처 간격 (초)
CAPTURE_COUNT = 5     # 캡처 횟수
OLLAMA_MODEL = "gemma3:1b" # 사용할 Ollama 모델

# --- 경로 설정 (중요: 상대 경로 사용) ---
# 이 스크립트 파일(app.py)이 있는 디렉토리를 기준으로 경로를 설정합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_FOLDER = os.path.join(BASE_DIR, "captures")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_skin_disease_model.pth")
FONT_PATH = os.path.join(BASE_DIR, "font", "NanumGothic.ttf")
CAMERA_INDEX = 0 # 라즈베리파이 카메라는 보통 0번입니다.

# --- 클래스 및 모델 설정 ---
class_names_kr = [
    '광선각화증', '기저세포암', '멜라닌세포모반', '보웬병', '비립종',
    '사마귀', '악성흑색종', '지루각화증', '편평세포암', '표피낭종',
    '피부섬유종', '피지샘증식종', '혈관종', '화농육아종', '흑색점'
]

class SkinDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super(SkinDiseaseClassifier, self).__init__()
        self.backbone = models.resnet101(weights=None) # weights=None으로 명시
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# --- Ollama Gemma3 함수 ---
def get_solution_from_gemma(disease_name):
    prompt = f"""당신은 피부 건강 전문 AI 어시스턴트입니다.
    다음 피부 질환에 대해 일반인이 이해하기 쉽게 설명하고, 집에서 할 수 있는 관리 방법과 전문적인 치료 방법을 단계별로 명확하게 구분해서 한국어로 알려주세요.

    **질환명: {disease_name}**
    """
    print(f"\n[{OLLAMA_MODEL} 모델에게 조언을 요청합니다...]")
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Ollama 모델 호출 중 오류: {e}\nOllama가 실행 중인지, 모델({OLLAMA_MODEL})이 다운로드되었는지 확인하세요."

# --- 메인 로직 ---
def main():
    if not os.path.exists(CAPTURE_FOLDER):
        os.makedirs(CAPTURE_FOLDER)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = torch.device('cpu')
    classification_model = SkinDiseaseClassifier(num_classes=15)
    
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 분류 모델 파일을 찾을 수 없습니다! 경로를 확인하세요: {MODEL_PATH}")
        return
    classification_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    classification_model.eval()

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except IOError:
        print(f"경고: 한글 폰트 파일을 찾을 수 없습니다: {FONT_PATH}. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"오류: 카메라({CAMERA_INDEX}번)를 열 수 없습니다.")
        return

    print("\n카메라 준비 완료. 'c'를 누르면 진단 시작, 'q'를 누르면 종료.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        crop_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

        input_tensor = transform(crop_frame).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = classification_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()

        label = f"{class_names_kr[predicted_class_idx]} ({confidence*100:.1f}%)"
        
        img_pil = Image.fromarray(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 10), "실시간 예측:", font=font, fill=(0, 255, 0))
        draw.text((10, 35), label, font=font, fill=(0, 255, 0))
        display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('Skin Diagnosis (Raspberry Pi)', display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            print("\n" + "="*40 + "\n진단을 시작합니다...")
            captured_classes = []
            for i in range(CAPTURE_COUNT):
                time.sleep(CAPTURE_INTERVAL)
                
                current_input_tensor = transform(crop_frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    current_outputs = classification_model(current_input_tensor)
                    current_predicted_idx = torch.argmax(torch.softmax(current_outputs, dim=1), dim=1).item()
                
                predicted_name = class_names_kr[current_predicted_idx]
                captured_classes.append(predicted_name)
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                capture_path = os.path.join(CAPTURE_FOLDER, f"capture_{timestamp}_{i+1}.png")
                cv2.imwrite(capture_path, crop_frame)
                print(f"촬영 {i+1}/{CAPTURE_COUNT}... 예측: {predicted_name}")

            print("\n" + "-"*40)
            if len(set(captured_classes)) == 1:
                final_diagnosis = captured_classes[0]
                print(f"최종 진단 결과: **{final_diagnosis}**")
                print("-" * 40)
                
                solution = get_solution_from_gemma(final_diagnosis)
                print("\n[Ollama Gemma3의 건강 조언]")
                print(solution)
                print("\n(주의: 이 정보는 참고용이며, 정확한 진단과 치료를 위해 반드시 전문 의료기관을 방문하세요.)")
            else:
                print("진단 실패: 예측 결과가 일치하지 않습니다.")
                print(f"지난 {CAPTURE_COUNT}번의 예측: {captured_classes}")
            
            print("="*40 + "\n다시 진단하려면 'c'를, 종료하려면 'q'를 누르세요.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

```