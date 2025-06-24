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

