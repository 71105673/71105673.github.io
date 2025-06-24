---
title: "Open CV" 
date: "2025-06-24"
thumbnail: "../../../assets/img/ARM/AI/image copy.png"
---

# Open-source Computer Vision

## 1. Ubuntu 설정

python3 -m venv .env
source .env/bin/activate

**Python 가상 환경에서 설치**

pip install opencv-python
pip install opencv-contrib-python

**설치 검증**

python3
import cv2
import numpy as np
print("cv ver:", cv2.__version__, "np ver:", np.__version__)

## 2. OpenCV란?

**Stitiching** : 여러 장의 이미지를 이어 붙여서 하나의 파노라마 이미지를 만드는 기술입니다. 

**Cuda** : OpenCV에서 CUDA를 활용한 Stitching은 GPU 가속을 통해 이미지 스티칭 속도를 향상시키는 기능입니다. 특히 고해상도 이미지나 여러 장의 사진을 빠르게 파노라마로 합칠 때 유용

**Color 조작 사이트**

HSL -> 색, 명도, 채도 확인 가능
http://www.w3schools.com/colors/ 

---

## 3. 실습

해당 코드를 통해 이미지 정보를 확인할 수 있다.

img.shape는 이미지의 크기 정보를 담고 있는 속성

```python
import numpy as np
import cv2

# 이미지 파일을 Read
img = cv2.imread("my_input.jpg")

# Image 란 이름의 Display 창 생성
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# Numpy ndarray K?W?C order
print(img.shape)

# Read 한 이미지 파일을 Display
cv2.imshow("image", img)

# 별도 키 입력이 있을 때 까지 대기
key = cv2.waitKey(0)

if key == ord('s'):  # s 키를 누르면 저장
    cv2.imwrite("output.png", img)
else:
    print("저장하지 않았습니다.")

# output.png 로 읽은 이미지 파일을 저장
cv2.imwrite("output.png", img)

# Destroy all windows
cv2.distroyAllWindows()
```
---

**여기서 RGB -> HSV 색상 영역으로 변경**

*HSV color sapce의 장점:*

1. 색상 기반 필터링/추적에 탁월

2. 조명 변화에 강함 (RGB는 밝기 변화에 취약)


```python
import numpy as np
import cv2

# 이미지 파일을 Read 하고 Color space 정보 출력
color = cv2.imread("my_input.jpg", cv2.IMREAD_COLOR)
#color = cv2.imread("strawberry_dark.jpg", cv2.IMREAD_COLOR)
print(color.shape)

height,width,channels = color.shape
cv2.imshow("Original Image", color)
cv2.imwrite("Original Image Output.png", color)

# Color channel 을 B,G,R 로 분할하여 출력
b,g,r = cv2.split(color)
rgb_split = np.concatenate((b,g,r), axis=1)
cv2.imshow("BGR Channels", rgb_split)
cv2.imwrite("BGR Channels Output.png",rgb_split)

# 색공간을 BGR 에서 HSV 로 변환
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

# Channel 을 H,S,V 로 분할하여 출력
h,s,v = cv2.split(hsv)
hsv_split = np.concatenate((h,s,v), axis=1)
cv2.imshow("Split HSV", hsv_split)
cv2.imwrite("Split HSV Output.png",rgb_split)

# [퀴즈 3번] HSV → RGB로 변환해서 출력 (주의: HSV2BGR 아님)
rgb_from_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
cv2.imshow("HSV to RGB", rgb_from_hsv)
cv2.imwrite("HSV to RGB Output.png",rgb_split)

# [퀴즈 4번] RGB → Grayscale 변환해서 출력
gray = cv2.cvtColor(rgb_from_hsv, cv2.COLOR_RGB2GRAY)
cv2.imshow("Grayscale from RGB", gray)
cv2.imwrite("Grayscale from RGB Output.png",rgb_split)

# 종료 대기
cv2.waitKey(0)
cv2.destroyAllWindows()

```
**출력 이미지 결과**

Original image
![alt text](<../../../assets/img/ARM/AI/Original Image Output.png>)

BGR Channels
![alt text](<../../../assets/img/ARM/AI/BGR Channels Output.png>)

Split HSV
![alt text](<../../../assets/img/ARM/AI/Split HSV Output.png>)

HSV to RGB
![alt text](<../../../assets/img/ARM/AI/HSV to RGB Output.png>)

Grayscale
![alt text](<../../../assets/img/ARM/AI/Grayscale from RGB Output.png>)

---

**이미지 Crop / Resize**

```python
import numpy as np
import cv2

# 이미지 파일을 Read
img = cv2.imread("my_input.jpg")

# Crop 200x100 from original image from (50, 25)=(x,y)
cropped = img[25:125, 50:250]

# Resize cropped image from 300x400 to 400x200
resized = cv2.resize(cropped, (400, 200))

# Display all
cv2.imshow("Original", img)
cv2.imshow("Cropped image", cropped)
cv2.imwrite("Cropped image output.png", cropped)
cv2.imshow("Resized image", resized)
cv2.imwrite("Resized image output.png", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
**출력 이미지 결과**

Original
![alt text](<../../../assets/img/ARM/AI/Original Image Output.png>)

Cropped (x: 25 ->250까지 200, y: 25 -> 125까지 100)

![alt text](<../../../assets/img/ARM/AI/Cropped image output.png>)

Resized
![alt text](<../../../assets/img/ARM/AI/Resized image output.png>)

---

**Reverse Image**
```python
import numpy as np
import cv2

src = cv2.imread("my_input.jpg", cv2.IMREAD_COLOR)
dst = cv2.bitwise_not(src)

cv2.imshow("src",src)
cv2.imwrite("src_output.png",src)

cv2.imshow("dst",dst)
cv2.imwrite("dst_output.png",dst)

cv2.waitKey()
cv2.destroyAllWindows()               
```

src

![alt text](../../../assets/img/ARM/AI/src_output.png)

dst(reverse)

![alt text](../../../assets/img/ARM/AI/dst_output.png)

---

**Binary(이진화)**

```python
import numpy as np
import cv2

src = cv2.imread("MR.jpeg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("dst",dst)
cv2.imwrite("MR_dst_out.png",dst)
cv2.waitKey()
cv.destroyAllWindows()
```

원본 파일

![alt text](../../../assets/img/ARM/AI/MR.jpeg)

이진화

![alt text](../../../assets/img/ARM/AI/MR_dst_out.png)

---

**Blur**

(9,9) -> 9X9 픽셀을 잡아서 평균 값을 냄, 값이 작을수록 블러가 강함

(1,1) -> 9X9 dptj 1,1을 중심점으로 (좌측 상단) 커널을 측정
(-1,-1) -> 자동 중심을 설정, 9X9는 9/2 ~ 4 로 설정

```python
import cv2

src = cv2.imread("my_input.jpg", cv2.IMREAD_COLOR)
dst = cv2.blur(src, (9, 9), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()
```
원본

![alt text](../../../assets/img/ARM/AI/MR.jpeg)

블러

![alt text](../../../assets/img/ARM/AI/blur_out.png)

---

****