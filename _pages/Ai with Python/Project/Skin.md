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



















# 🏥 ONNX 기반 피부 질환 진단 시스템

H5 모델을 8비트 양자화하고 ONNX/TFLite로 변환하여 더 빠르고 효율적인 피부 질환 진단 시스템입니다.

## 📋 시스템 구성

```
📁 onnx_skin_diagnosis/
├── convert_h5_to_onnx.py          # ✅ H5 → ONNX 변환 스크립트
├── camera_h5_diagnosis.py         # 📱 H5 모델 카메라 진단 프로그램
├── camera_onnx_diagnosis.py       # 📱 기본 ONNX 카메라 진단 프로그램
├── camera_onnx_optimized.py       # 🚀 최적화된 ONNX 카메라 진단 프로그램
├── README.md                      # 📖 프로젝트 설명서
├── README_WINDOW.md               # 📖 Windows 버전 설명서
├── OPTIMIZED_option.md            # 📖 최적화 옵션 설명서
├── requirements.txt               # 📋 필요한 패키지 목록
├── 📁 model/                      # 🧠 모델 저장소
│   ├── skin_model.h5              # 원본 Keras 모델 (11.4MB)
│   ├── skin_model.onnx            # 변환된 ONNX 모델 (9.6MB)
│   ├── skin_model_quantized.tflite # 양자화 TFLite 모델 (2.9MB)
│   ├── skin_model_quantized_dynamic.onnx # 동적 양자화 ONNX (2.6MB)
│   └── skin_model_quantized_static.onnx  # 정적 양자화 ONNX (2.6MB)
└── 📁 captures/                   # 📸 진단 이미지 저장 폴더
```

## 🚀 설치 및 설정

<details>
<summary> # window </summary>
<div markdown="1">

### 1. 필요한 패키지 설치

```bash
# 기본 패키지
pip install -r requirements.txt

## 📂 사용 방법

### 1단계: 모델 변환

먼저 H5 모델을 ONNX/TFLite로 변환합니다:

```bash
python3 convert_h5_to_onnx.py
```

**변환 결과:**
- ✅ `skin_model.onnx` - ONNX 모델 생성
- ✅ `skin_model_quantized.tflite` - 8비트 양자화 TFLite 모델 생성
- ✅ `skin_model_quantized_dynamic.onxx` - 8비트 동적 양자화 onxx 모델 생성
- ✅ `skin_model_static_dynamic.onxx` - 8비트 정적 양자화 onxx 모델 생성
 

### 2단계: 진단 프로그램 실행

```bash
# ollama 실행 (터미널 하나 더 열어서 진행행)
ollama run gemma3:1b

```

```bash
# h5 기본
python camera_h5_diagnosis.py

# onxx 기본
python camera_onnx_diagnosis.py

# onxx runtime 적용
python camera_onnx_optimized.py

```
</div>
</details>


<details>
<summary> # Linux </summary>
<div markdown="1">


### 1. 필요한 패키지 설치

```bash
# 기본 패키지 설치
pip install -r requirements.txt

# Pillow 최신 버전 업그레이드 (텍스트 렌더링 오류 방지용)
pip install --upgrade pillow

# 리눅스(Ubuntu) 환경에서 한글 폰트가 깨질 경우 아래 명령어로 나눔글꼴 설치
sudo apt update
sudo apt install fonts-nanum

💡fonts-nanum은 한글을 깨지지 않게 표시하기 위해 필요합니다. 설치 후 코드에서 다음과 같이 경로를 설정하세요:
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# Ollama 설치 (Snap 기반)
sudo snap install ollama

## 📂 사용 방법

### 1단계: 모델 변환

먼저 H5 모델을 ONNX/TFLite로 변환합니다:


```bash
python3 convert_h5_to_onnx.py
```


**변환 결과:**
- ✅ `skin_model.onnx` - ONNX 모델 생성
- ✅ `skin_model_quantized.tflite` - 8비트 양자화 TFLite 모델 생성
- ✅ `skin_model_quantized_dynamic.onxx` - 8비트 동적 양자화 onxx 모델 생성
- ✅ `skin_model_static_dynamic.onxx` - 8비트 정적 양자화 onxx 모델 생성
 


### 2단계: 진단 프로그램 실행

```bash
# ollama 실행 (터미널 하나 더 열어서 진행행)
ollama run gemma3:1b

```

```bash
# h5 기본
python3 camera_h5_diagnosis.py

# onxx 기본
python3 camera_onnx_diagnosis.py

# onxx runtime 적용
python3 camera_onnx_optimized.py

```

</div>
</details>




## 🎯 모델 우선순위

프로그램은 다음 순서로 모델을 로드합니다:

1. **ONNX 모델** (최우선) - 가장 빠른 추론 속도
2. **TFLite 모델** (8비트 양자화) - 작은 파일 크기, 빠른 속도
3. **원본 H5 모델** (백업) - 변환 실패 시 사용

## 📊 진단 클래스 (7개)

1. **기저세포암** - 가장 흔한 피부암
2. **표피낭종** - 양성 낭종
3. **혈관종** - 혈관 증식 병변 
4. **비립종** - 작은 각질 주머니
5. **정상피부** - 건강한 피부
6. **편평세포암** - 두 번째 흔한 피부암
7. **사마귀** - HPV 감염

## 🎮 조작 방법

### 실시간 모드
- **카메라 화면**: 실시간 예측 결과 표시
- **모델 정보**: 사용 중인 모델 타입 (ONNX/TFLite/H5) 표시

### 진단 모드
- **'c' 키**: 5초간 연속 촬영하여 정확한 진단 시작
- **진단 과정**: 5번 촬영 → 결과 일치 확인 → 최종 진단
- **AI 조언**: Ollama Gemma3를 통한 개인맞춤 건강 조언

### 기타
- **'q' 키**: 프로그램 종료

## 🔧 성능 최적화

### 모델 크기 비교
- **원본 H5**: ~50MB
- **ONNX**: ~45MB (10% 감소)
- **TFLite 양자화**: ~12MB (75% 감소)

### 추론 속도
- **ONNX**: 가장 빠름 (CPU 최적화)
- **TFLite**: 빠름 (메모리 효율적)
- **H5**: 보통 (TensorFlow 오버헤드)

## 🛠️ 문제 해결

### 모델 로드 실패
```bash
❌ ONNX 모델 로드 실패: ...
❌ TFLite 모델 로드 실패: ...
✅ 원본 H5 모델을 사용합니다.
```
→ 변환 과정을 다시 실행하세요.

### 카메라 접근 실패
```bash
❌ 카메라를 열 수 없습니다.
```
→ 다른 프로그램이 카메라를 사용 중인지 확인하세요.

### Ollama 연결 실패
```bash
Ollama 모델을 호출하는 중 오류가 발생했습니다...
```
→ `ollama run gemma3` 명령으로 모델을 실행하세요.

## 📈 추가 기능

### 이미지 저장
- 진단 시 모든 캡처 이미지가 `captures/` 폴더에 저장됩니다
- 파일명: `capture_YYYYMMDD_HHMMSS_N.png`

### AI 건강 조언
- Ollama Gemma3 모델을 통한 실시간 건강 조언 제공
- 간결하고 실용적인 정보 제공 (200자 내외)

## ⚠️ 주의사항

1. **의학적 조언 아님**: 이 시스템은 참고용이며, 정확한 진단은 전문의와 상담하세요.
2. **조명 조건**: 충분한 조명에서 사용하세요.
3. **카메라 위치**: 진단 부위를 화면 중앙에 위치시키세요.
4. **정확성**: 5번 연속 촬영에서 같은 결과가 나와야 신뢰할 수 있습니다.

## 📞 기술 지원

문제가 발생하면 다음 사항을 확인해주세요:

1. **Python 버전**: 3.8 이상 권장
2. **패키지 버전**: 최신 버전 사용 권장
3. **모델 파일**: 변환된 모델 파일이 정상적으로 생성되었는지 확인
4. **하드웨어**: 충분한 RAM과 CPU 성능 필요

---

**© 2024 ONNX 기반 피부 질환 진단 시스템** 

## requirements.txt
```
# 최적화된 ONNX 기반 피부 질환 진단 시스템 필수 패키지

# 머신러닝 및 딥러닝
tensorflow>=2.10.0
numpy>=1.21.0

# ONNX 관련 (최적화 버전)
onnxruntime>=1.16.0
onnxruntime-gpu>=1.16.0  # GPU 지원 (CUDA/DirectML)
tf2onnx>=1.14.0
onnx>=1.14.0

# 컴퓨터 비전
opencv-python>=4.7.0
Pillow>=9.0.0

# 시스템 모니터링 및 최적화
psutil>=5.9.0

# AI 건강 조언 (선택사항)
ollama>=0.1.0

# 성능 최적화 (선택사항)
# onnxruntime-openvino  # Intel OpenVINO 지원
# onnxruntime-directml  # DirectML 지원 (Windows) 
```

## Diagnosis.py
```python
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import time
import os
import ollama
import onnxruntime as ort

# --- 설정 ---
CAPTURE_INTERVAL = 1  # 캡처 간격 (초)
CAPTURE_COUNT = 5     # 캡처 횟수
CAPTURE_FOLDER = "captures" # 캡처 이미지 저장 폴더
OLLAMA_MODEL = "gemma3:1b-it-qat" # 사용할 Ollama 모델

# 모델 경로 설정
ONNX_MODEL_PATH = "./model/skin_model.onnx"
TFLITE_MODEL_PATH = "./model/skin_model_quantized.tflite"

# --- 클래스 및 모델 설정 ---
# 클래스명 (7개 클래스)
class_names_kr = [
    '기저세포암',
    '표피낭종',
    '혈관종',
    '비립종',
    '정상피부',
    '편평세포암',
    '사마귀'
]


class ONNXModel:
    def __init__(self, model_path):
        """ONNX 모델 로드"""
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"ONNX 모델 로드 성공: {model_path}")
    
    def predict(self, input_data):
        """ONNX 모델 예측"""
        result = self.session.run([self.output_name], {self.input_name: input_data})
        return result[0]

# --- TFLite 모델 클래스 ---
class TFLiteModel:
    def __init__(self, model_path):
        """TFLite 모델 로드"""
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 입력 및 출력 텐서 정보
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"TFLite 모델 로드 성공: {model_path}")
    
    def predict(self, input_data):
        """TFLite 모델 예측"""
        # 입력 데이터 설정
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # 추론 실행
        self.interpreter.invoke()
        
        # 출력 데이터 가져오기
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

# --- Ollama Gemma3 함수 ---
def get_solution_from_gemma(disease_name):
    """
    Ollama를 통해 로컬 Gemma3 모델을 호출하여 질환에 대한 해결책을 생성합니다.
    """
    prompt = f"""당신은 피부 건강 전문 AI 어시스턴트입니다.
다음 피부 질환에 대해 일반인이 이해하기 쉽게 설명하고, 집에서 할 수 있는 관리 방법과 전문적인 치료 방법을 단계별로 명확하게 구분해서 한국어로 알려주세요.

**질환명: {disease_name}**

다음 형식으로 답변해주세요:
1. 질환 설명 (간단명료하게)
2. 즉시 조치사항 (응급성 여부)
3. 가정 관리 방법
4. 전문 치료 방법
5. 주의사항

200자 내외로 간결하게 답변해주세요."""
    
    print(f"\n[{OLLAMA_MODEL} 모델에게 조언을 요청합니다... 잠시만 기다려 주세요.]")
    
    try:
        # Ollama 서버와 통신
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Ollama 모델을 호출하는 중 오류가 발생했습니다: {e}\nOllama가 실행 중인지 확인하세요. 터미널에서 'ollama run {OLLAMA_MODEL}' 명령을 실행해야 할 수 있습니다."

# --- 모델 초기화 함수 ---
def initialize_model():
    """사용 가능한 모델을 초기화합니다."""
    model = None
    model_type = None
    
    # 1. ONNX 모델 시도
    if os.path.exists(ONNX_MODEL_PATH):
        try:
            model = ONNXModel(ONNX_MODEL_PATH)
            model_type = "ONNX"
            print("ONNX 모델을 사용합니다.")
        except Exception as e:
            print(f"ONNX 모델 로드 실패: {e}")
    
    # 2. TFLite 모델 시도 (ONNX 실패 시)
    if model is None and os.path.exists(TFLITE_MODEL_PATH):
        try:
            model = TFLiteModel(TFLITE_MODEL_PATH)
            model_type = "TFLite"
            print("TFLite 모델을 사용합니다.")
        except Exception as e:
            print(f"TFLite 모델 로드 실패: {e}")
    
    # 3. 원본 H5 모델 시도 (둘 다 실패 시)
    if model is None:
        try:
            import tensorflow as tf
            from tensorflow import keras
            h5_model_path = "C:/Users/kccistc/project/pth/jaehong_skin_model.h5"
            model = keras.models.load_model(h5_model_path)
            model_type = "H5"
            print("원본 H5 모델을 사용합니다.")
        except Exception as e:
            print(f"H5 모델 로드 실패: {e}")
    
    return model, model_type

# --- 메인 로직 ---
def main():
    print("ONNX Skin Diagnosis System")
    print("=" * 50)
    
    # 캡처 폴더 생성
    if not os.path.exists(CAPTURE_FOLDER):
        os.makedirs(CAPTURE_FOLDER)
    
    # 모델 초기화
    model, model_type = initialize_model()
    if model is None:
        print("사용 가능한 모델이 없습니다.")
        print("먼저 convert_h5_to_onnx.py를 실행하여 모델을 변환하세요.")
        return
    
    print(f"사용 중인 모델: {model_type}")
    
    # 폰트 설정
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothicExtraBold.ttf"
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        print(f"오류: 폰트 파일을 찾을 수 없습니다: {font_path}. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    # 카메라 설정
    cap = cv2.VideoCapture(1) # 외부 웹캠
    if not cap.isOpened():
        cap = cv2.VideoCapture(0) # 내장 웹캠
        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

    print("카메라가 준비되었습니다.")
    print("화면을 보며 진단할 부위를 중앙에 위치시키세요.")
    print("키보드 'c'를 누르면 5초간 연속으로 촬영하여 진단합니다.")
    print("키보드 'q'를 누르면 프로그램을 종료합니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("오류: 카메라에서 프레임을 읽을 수 없습니다.")
            break

        # 중앙 1:1 영역 crop
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        crop_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

        # --- 실시간 예측 ---
        # 이미지 전처리
        img_array = cv2.resize(crop_frame, (96, 96))
        img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
        img_array = img_array.astype(np.float32) / 255.0  # 정규화

        # 모델 타입에 따른 예측
        if model_type == "H5":
            predictions = model.predict(img_array, verbose=0)
        else:
            predictions = model.predict(img_array)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]

        # 결과 텍스트 생성
        if predicted_class_idx < len(class_names_kr):
            label = f"{class_names_kr[predicted_class_idx]} ({confidence*100:.1f}%)"
        else:
            label = f"알 수 없음 ({confidence*100:.1f}%)"
 
        # 화면에 표시 (Pillow 사용)
        img_pil = Image.fromarray(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 10), f"실시간 예측 ({model_type}):", font=font, fill=(0, 255, 0))
        draw.text((10, 35), label, font=font, fill=(0, 255, 0))
        display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('ONNX Skin Disease Diagnosis', display_frame)

        key = cv2.waitKey(1) & 0xFF

        # --- 'c' 키를 눌러 연속 캡처 및 진단 ---
        if key == ord('c'):
            # 화면을 검게 만들고 "의사의 답변 준비중..." 메시지 표시
            black_screen = np.zeros_like(display_frame)
            
            # Pillow를 사용하여 텍스트 추가
            img_pil_black = Image.fromarray(cv2.cvtColor(black_screen, cv2.COLOR_BGR2RGB))
            draw_black = ImageDraw.Draw(img_pil_black)
            
            text = "의사의 답변 준비중..."
            
            # 텍스트 크기 계산
            try:
                # Pillow 10.0.0 이상
                text_bbox = draw_black.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                # 이전 버전의 Pillow
                text_width, text_height = draw_black.textsize(text, font=font)

            text_x = (black_screen.shape[1] - text_width) // 2
            text_y = (black_screen.shape[0] - text_height) // 2
            
            draw_black.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
            
            # OpenCV 형식으로 다시 변환하여 표시
            black_screen_with_text = cv2.cvtColor(np.array(img_pil_black), cv2.COLOR_RGB2BGR)
            cv2.imshow('ONNX Skin Disease Diagnosis', black_screen_with_text)
            cv2.waitKey(1) # 화면을 즉시 업데이트

            print("\n" + "="*40)
            print(f"진단을 시작합니다. {CAPTURE_COUNT}초 동안 {CAPTURE_COUNT}번 촬영합니다.")
            print("="*40)
            
            captured_classes = []
            
            for i in range(CAPTURE_COUNT):
                time.sleep(CAPTURE_INTERVAL)
                
                # 현재 프레임(crop_frame)으로 예측
                current_img_array = cv2.resize(crop_frame, (96, 96))
                current_img_array = np.expand_dims(current_img_array, axis=0)
                current_img_array = current_img_array.astype(np.float32) / 255.0

                # 모델 타입에 따른 예측
                if model_type == "H5":
                    current_predictions = model.predict(current_img_array, verbose=0)
                else:
                    current_predictions = model.predict(current_img_array)
                
                current_predicted_idx = np.argmax(current_predictions[0])
                
                predicted_name = class_names_kr[current_predicted_idx]
                captured_classes.append(predicted_name)
                
                # 캡처 이미지 저장
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                capture_path = os.path.join(CAPTURE_FOLDER, f"capture_{timestamp}_{i+1}.png")
                cv2.imwrite(capture_path, crop_frame)
                
                print(f"촬영 {i+1}/5... 예측: {predicted_name} (이미지 저장: {capture_path})")

            # --- 최종 진단 ---
            print("\n" + "-"*40)
            if len(set(captured_classes)) == 1:
                final_diagnosis = captured_classes[0]
                print(f"최종 진단 결과: **{final_diagnosis}**")
                print(f"사용 모델: {model_type}")
                print("-"*40)
                
                # Gemma3 해결책 요청
                solution = get_solution_from_gemma(final_diagnosis)
                print("\n[Ollama Gemma3의 건강 조언]")
                print(solution)
                print("\n(주의: 이 정보는 참고용이며, 정확한 진단과 치료를 위해 반드시 전문 의료기관을 방문하세요.)")
                
            else:
                print("진단 실패: 예측 결과가 일치하지 않습니다.")
                print(f"지난 {CAPTURE_COUNT}번의 예측: {captured_classes}")
            
            print("="*40)
            print("\n다시 진단하려면 'c'를, 종료하려면 'q'를 누르세요.")

        # --- 'q' 키를 눌러 종료 ---
        elif key == ord('q'):
            print("프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
```

## Optimized.py
```python
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import time
import os
import ollama
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import psutil
import threading
import queue

# --- 설정 ---
CAPTURE_INTERVAL = 1  # 캡처 간격 (초)
CAPTURE_COUNT = 5     # 캡처 횟수
CAPTURE_FOLDER = "captures" # 캡처 이미지 저장 폴더
OLLAMA_MODEL = "gemma3" # 사용할 Ollama 모델

# 모델 경로 설정

ONNX_MODEL_PATH = "./model/skin_model.onnx"
ONNX_OPTIMIZED_PATH = "./model/skin_model_optimized.onnx"
ONNX_QUANTIZED_PATH = "./model/skin_model_quantized.onnx"
TFLITE_MODEL_PATH = "./model/skin_model_quantized.tflite"

# --- 클래스 및 모델 설정 ---
class_names_kr = [
    '기저세포암',
    '보웬병',
    '표피낭종',
    '비립종',
    '정상피부',
    '화농성 육아종',
    '편평세포암',
    '사마귀'
]

# --- 최적화된 ONNX 모델 클래스 ---
class OptimizedONNXModel:
    def __init__(self, model_path, optimization_level="all", use_gpu=False):
        """
        최적화된 ONNX 모델 로드
        
        Args:
            model_path: 모델 경로
            optimization_level: 최적화 레벨 ("disable", "basic", "extended", "all")
            use_gpu: GPU 사용 여부
        """
        self.model_path = model_path
        self.optimization_level = optimization_level
        self.use_gpu = use_gpu
        
        # 세션 옵션 설정
        self.session_options = ort.SessionOptions()
        
        # 최적화 레벨 설정
        if optimization_level == "disable":
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        elif optimization_level == "basic":
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif optimization_level == "extended":
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:  # "all"
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 병렬 처리 설정
        cpu_count = psutil.cpu_count(logical=False)
        self.session_options.intra_op_num_threads = cpu_count
        self.session_options.inter_op_num_threads = cpu_count
        
        # 메모리 패턴 최적화
        self.session_options.enable_mem_pattern = True
        self.session_options.enable_cpu_mem_arena = True
        
        # 실행 제공자 설정
        providers = self._get_providers()
        
        try:
            # ONNX 세션 생성
            self.session = ort.InferenceSession(
                model_path, 
                sess_options=self.session_options,
                providers=providers
            )
            
            # 입출력 정보
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # 모델 정보 출력
            print(f" 최적화된 ONNX 모델 로드 성공: {model_path}")
            print(f"    최적화 레벨: {optimization_level}")
            print(f"    Intra-op threads: {self.session_options.intra_op_num_threads}")
            print(f"    Inter-op threads: {self.session_options.inter_op_num_threads}")
            print(f"    사용 중인 Providers: {self.session.get_providers()}")
            
        except Exception as e:
            print(f" 최적화된 ONNX 모델 로드 실패: {e}")
            raise
    
    def _get_providers(self):
        """사용 가능한 실행 제공자 반환"""
        providers = []
        
        # GPU 사용 시도
        if self.use_gpu:
            # DirectML (Windows)
            if 'DmlExecutionProvider' in ort.get_available_providers():
                providers.append('DmlExecutionProvider')
                print(" DirectML Provider 사용 (Windows GPU)")
            
            # CUDA (NVIDIA)
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
                print(" CUDA Provider 사용 (NVIDIA GPU)")
            
            # OpenVINO (Intel)
            if 'OpenVINOExecutionProvider' in ort.get_available_providers():
                providers.append('OpenVINOExecutionProvider')
                print(" OpenVINO Provider 사용 (Intel GPU)")
        
        # CPU는 항상 백업으로 추가
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def predict(self, input_data):
        """최적화된 예측"""
        try:
            result = self.session.run([self.output_name], {self.input_name: input_data})
            return result[0]
        except Exception as e:
            print(f" 예측 실패: {e}")
            return None

# --- 동적 양자화 함수 ---
def create_quantized_model(onnx_model_path, quantized_model_path):
    """동적 양자화된 ONNX 모델 생성"""
    try:
        print(" 동적 양자화 시작...")
        
        # 동적 양자화 실행
        quantized_model = quantize_dynamic(
            onnx_model_path,
            quantized_model_path,
            weight_type=QuantType.QUInt8,
            optimize_model=True
        )
        
        # 파일 크기 비교
        original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)
        
        print(f" 동적 양자화 완료")
        print(f"    크기 변화: {original_size:.2f}MB → {quantized_size:.2f}MB ({quantized_size/original_size*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f" 동적 양자화 실패: {e}")
        return False

# --- 성능 벤치마킹 함수 ---
def benchmark_model(model, test_data, num_runs=100):
    """모델 성능 벤치마킹"""
    print(f" 성능 벤치마킹 시작 ({num_runs}회 실행)...")
    
    # 워밍업
    for _ in range(10):
        model.predict(test_data)
    
    # 실제 벤치마킹
    start_time = time.time()
    for _ in range(num_runs):
        model.predict(test_data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    fps = 1 / (avg_time / 1000)
    
    print(f"    평균 추론 시간: {avg_time:.2f}ms")
    print(f"    초당 프레임: {fps:.1f} FPS")
    
    return avg_time, fps

# --- 비동기 예측 클래스 ---
class AsyncPredictor:
    def __init__(self, model):
        self.model = model
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.prediction_thread = threading.Thread(target=self._prediction_worker)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
        self.last_prediction = None
    
    def _prediction_worker(self):
        """백그라운드 예측 작업자"""
        while True:
            try:
                input_data = self.input_queue.get(timeout=0.1)
                result = self.model.predict(input_data)
                
                # 결과 큐가 가득 찬 경우 이전 결과 제거
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.output_queue.put(result)
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f" 비동기 예측 오류: {e}")
    
    def predict_async(self, input_data):
        """비동기 예측 요청"""
        # 입력 큐가 가득 찬 경우 이전 요청 제거
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            self.input_queue.put_nowait(input_data)
        except queue.Full:
            pass
    
    def get_prediction(self):
        """예측 결과 가져오기"""
        try:
            result = self.output_queue.get_nowait()
            self.last_prediction = result
            return result
        except queue.Empty:
            return self.last_prediction

# --- 모델 초기화 함수 ---
def initialize_optimized_model():
    """최적화된 모델 초기화"""
    print(" 최적화된 ONNX 모델 초기화 시작...")
    
    # 1. 원본 ONNX 모델 확인
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f" 원본 ONNX 모델이 없습니다: {ONNX_MODEL_PATH}")
        print(" 먼저 convert_h5_to_onnx.py를 실행하여 모델을 변환하세요.")
        return None, None
    
    # 2. 동적 양자화 모델 생성 (없는 경우에만)
    if not os.path.exists(ONNX_QUANTIZED_PATH):
        print(" 동적 양자화 모델 생성 중...")
        create_quantized_model(ONNX_MODEL_PATH, ONNX_QUANTIZED_PATH)
    
    # 3. 최적화된 모델 로드 시도
    models_to_try = [
        (ONNX_QUANTIZED_PATH, "최적화 + 동적 양자화"),
        (ONNX_MODEL_PATH, "기본 ONNX")
    ]
    
    for model_path, description in models_to_try:
        if os.path.exists(model_path):
            try:
                # GPU 사용 가능 여부 확인
                use_gpu = len([p for p in ort.get_available_providers() 
                              if p in ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']]) > 0
                
                model = OptimizedONNXModel(
                    model_path, 
                    optimization_level="all",
                    use_gpu=use_gpu
                )
                
                print(f" {description} 모델 로드 성공")
                return model, description
                
            except Exception as e:
                print(f" {description} 모델 로드 실패: {e}")
                continue
    
    # 4. 백업으로 H5 모델 시도
    try:
        import tensorflow as tf
        from tensorflow import keras
        h5_model_path = "C:/Users/kccistc/project/pth/jaehong_skin_model.h5"
        model = keras.models.load_model(h5_model_path)
        print(" 백업 H5 모델 로드 성공")
        return model, "H5 백업"
    except Exception as e:
        print(f" 백업 H5 모델 로드 실패: {e}")
    
    return None, None

# --- Ollama Gemma3 함수 ---
def get_solution_from_gemma(disease_name):
    """Ollama를 통해 로컬 Gemma3 모델을 호출하여 질환에 대한 해결책을 생성합니다."""
    prompt = f"""당신은 피부 건강 전문 AI 어시스턴트입니다.
다음 피부 질환에 대해 일반인이 이해하기 쉽게 설명하고, 집에서 할 수 있는 관리 방법과 전문적인 치료 방법을 단계별로 명확하게 구분해서 한국어로 알려주세요.

**질환명: {disease_name}**

다음 형식으로 답변해주세요:
1. 질환 설명 (간단명료하게)
2. 즉시 조치사항 (응급성 여부)
3. 가정 관리 방법
4. 전문 치료 방법
5. 주의사항

200자 내외로 간결하게 답변해주세요."""
    
    print(f"\n[{OLLAMA_MODEL} 모델에게 조언을 요청합니다... 잠시만 기다려 주세요.]")
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Ollama 모델을 호출하는 중 오류가 발생했습니다: {e}\nOllama가 실행 중인지 확인하세요."

# --- 메인 로직 ---
def main():
    print(" 최적화된 ONNX 기반 피부 질환 진단 시스템")
    print("=" * 55)
    
    # 시스템 정보 출력
    print(f" CPU 코어: {psutil.cpu_count(logical=False)} 물리 / {psutil.cpu_count(logical=True)} 논리")
    print(f" 사용 가능한 메모리: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f" 사용 가능한 ONNX Providers: {ort.get_available_providers()}")
    print("=" * 55)
    
    # 캡처 폴더 생성
    if not os.path.exists(CAPTURE_FOLDER):
        os.makedirs(CAPTURE_FOLDER)
    
    # 최적화된 모델 초기화
    model, model_type = initialize_optimized_model()
    if model is None:
        print(" 사용 가능한 모델이 없습니다.")
        return
    
    print(f" 사용 중인 모델: {model_type}")
    
    # 성능 벤치마킹 (ONNX 모델의 경우)
    if "ONNX" in model_type or "최적화" in model_type:
        test_data = np.random.random((1, 96, 96, 3)).astype(np.float32)
        avg_time, fps = benchmark_model(model, test_data)
        
        # 비동기 예측기 초기화
        async_predictor = AsyncPredictor(model)
        use_async = True
        print(" 비동기 예측 모드 활성화")
    else:
        use_async = False
        print(" 동기 예측 모드 사용")
    
    # 폰트 설정
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothicExtraBold.ttf"
    try:
        font = ImageFont.truetype(font_path, 20)
        small_font = ImageFont.truetype(font_path, 14)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()


    # 카메라 설정
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(" 카메라를 열 수 없습니다.")
            return

    # 카메라 해상도 최적화
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(" 카메라가 준비되었습니다.")
    print("화면을 보며 진단할 부위를 중앙에 위치시키세요.")
    print("키보드 'c'를 누르면 5초간 연속으로 촬영하여 진단합니다.")
    print("키보드 'q'를 누르면 프로그램을 종료합니다.")
    print("키보드 'b'를 누르면 벤치마킹을 다시 실행합니다.")

    # 성능 측정 변수
    frame_count = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("오류: 카메라에서 프레임을 읽을 수 없습니다.")
            break

        # 중앙 1:1 영역 crop
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        crop_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

        # 이미지 전처리
        img_array = cv2.resize(crop_frame, (96, 96))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype(np.float32) / 255.0

        # 예측 수행
        if use_async:
            # 비동기 예측
            async_predictor.predict_async(img_array)
            predictions = async_predictor.get_prediction()
            
            if predictions is not None:
                predicted_class_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_idx]
            else:
                predicted_class_idx = 0
                confidence = 0.0
        else:
            # 동기 예측
            if "ONNX" in model_type or "최적화" in model_type:
                predictions = model.predict(img_array)
                if predictions is not None:
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_idx]
                else:
                    predicted_class_idx = 0
                    confidence = 0.0
            else:
                predictions = model.predict(img_array, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_idx]

        # FPS 계산
        frame_count += 1
        if frame_count % 30 == 0:
            fps_end_time = time.time()
            current_fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
        else:
            current_fps = 0

        # 결과 텍스트 생성
        label = f"{class_names_kr[predicted_class_idx]} ({confidence*100:.1f}%)"
        
        # 화면에 표시
        img_pil = Image.fromarray(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 메인 정보
        draw.text((10, 10), f" 실시간 예측 ({model_type}):", font=font, fill=(0, 255, 0))
        draw.text((10, 35), label, font=font, fill=(0, 255, 0))
        
        # 성능 정보
        if frame_count % 30 == 0 and current_fps > 0:
            draw.text((10, 65), f" FPS: {current_fps:.1f}", font=small_font, fill=(255, 255, 0))
        
        # 사용 중인 Provider 정보 (ONNX 모델의 경우)
        if hasattr(model, 'session'):
            provider_info = model.session.get_providers()[0]
            draw.text((10, 85), f" Provider: {provider_info.replace('ExecutionProvider', '')}", font=small_font, fill=(255, 255, 0))
        
        display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow('최적화된 ONNX 피부 진단', display_frame)

        key = cv2.waitKey(1) & 0xFF

        # 'b' 키로 벤치마킹 실행
        if key == ord('b') and ("ONNX" in model_type or "최적화" in model_type):
            print("\n" + "="*50)
            print(" 실시간 벤치마킹 실행")
            print("="*50)
            avg_time, fps = benchmark_model(model, img_array)

        # 'c' 키로 진단 실행
        elif key == ord('c'):
            # 진단 로직 (기존과 동일)
            print("\n" + "="*40)
            print(f"진단을 시작합니다. {CAPTURE_COUNT}초 동안 {CAPTURE_COUNT}번 촬영합니다.")
            print("="*40)
            
            captured_classes = []
            
            for i in range(CAPTURE_COUNT):
                time.sleep(CAPTURE_INTERVAL)
                
                # 현재 프레임으로 예측
                if "ONNX" in model_type or "최적화" in model_type:
                    current_predictions = model.predict(img_array)
                    if current_predictions is not None:
                        current_predicted_idx = np.argmax(current_predictions[0])
                    else:
                        current_predicted_idx = 0
                else:
                    current_predictions = model.predict(img_array, verbose=0)
                    current_predicted_idx = np.argmax(current_predictions[0])
                
                predicted_name = class_names_kr[current_predicted_idx]
                captured_classes.append(predicted_name)
                
                # 캡처 이미지 저장
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                capture_path = os.path.join(CAPTURE_FOLDER, f"capture_{timestamp}_{i+1}.png")
                cv2.imwrite(capture_path, crop_frame)
                
                print(f"촬영 {i+1}/5... 예측: {predicted_name}")

            # 최종 진단
            print("\n" + "-"*40)
            if len(set(captured_classes)) == 1:
                final_diagnosis = captured_classes[0]
                print(f"최종 진단 결과: **{final_diagnosis}**")
                print(f"사용 모델: {model_type}")
                print("-"*40)
                
                # Gemma3 해결책 요청
                solution = get_solution_from_gemma(final_diagnosis)
                print("\n[Ollama Gemma3의 건강 조언]")
                print(solution)
                print("\n(주의: 이 정보는 참고용이며, 정확한 진단과 치료를 위해 반드시 전문 의료기관을 방문하세요.)")
                
            else:
                print("진단 실패: 예측 결과가 일치하지 않습니다.")
                print(f"지난 {CAPTURE_COUNT}번의 예측: {captured_classes}")
            
            print("="*40)
            print("\n다시 진단하려면 'c'를, 벤치마킹은 'b'를, 종료하려면 'q'를 누르세요.")

        # 'q' 키로 종료
        elif key == ord('q'):
            print("프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
```