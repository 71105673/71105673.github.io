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
