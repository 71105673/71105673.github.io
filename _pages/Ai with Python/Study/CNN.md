---
title: "Day-6 CNN" 
date: "2025-06-30"
thumbnail: "../../../assets/img/ARM/AI/image copy 32.png"
---

# CNN 이란?
***Convolutional Neural Network(합성곱 신경망)***
-이미지 인식, 영상 분석, 자연어 처리에 사용되는 딥러닝 모델 구조.

## 📌 CNN의 핵심 개념

| 구성 요소                              | 설명                                                     |
| ---------------------------------- | ------------------------------------------------------ |
| **Convolution Layer (합성곱층)**       | 필터(또는 커널)를 사용해 입력 이미지의 특징(Feature)을 추출함. 가장 핵심적인 층.    |
| **Activation Function (활성화 함수)**   | 주로 ReLU(Rectified Linear Unit)를 사용하여 비선형성을 추가함.        |
| **Pooling Layer (풀링층)**            | 특성 맵을 축소해 연산량 감소 및 중요한 특징 유지 (MaxPooling이 일반적).        |
| **Fully Connected Layer (완전 연결층)** | 추출된 특징을 바탕으로 분류나 예측 등을 수행. 마지막에 Softmax 또는 Sigmoid 사용. |

## 📊 CNN vs 일반 MLP (다층 퍼셉트론)

| 항목     | CNN                 | MLP       |
| ------ | ------------------- | --------- |
| 특징 추출  | 자동으로 이미지 특징 추출      | 수동 혹은 제한적 |
| 입력 구조  | 주로 이미지/2D 데이터       | 1차원 벡터    |
| 파라미터 수 | 상대적으로 적음 (공유 필터 사용) | 매우 많음     |

## CNN 모델의 구성

![alt text](../../../assets/img/ARM/AI/CNN/image.png)

>처음 입력은 32×32×3의 3차원 데이터이다. 
>
>한 번의 합성곱 층을 거쳐 28×28×6인 크기의 3차원 데이터가 되었다.

>CNN Layer에는 합성곱이 6번(6개의 필터) 사용된 것을 알 수 있다. 
>
>다음으로 Pooling layer를 거쳐 14×14×6인 크기의 데이터가 되었다.

>이것을 벡터화(Flatten)함으로써 1,176 크기의 1차원 벡터가 된다.
>
>마지막으로 FC layer를 거쳐 최종적으로 class 개수에 .맞도록 10개의 값을 갖는 출력층이 만들어진다.

### CNN 구성 요소 -> Convolutuon Layer 연산

![alt text](<../../../assets/img/ARM/AI/CNN/image copy.png>)

>합성곱은 텐서(tensor)와 텐서 사이에서 정의되는
연산이다.

>텐서는 차원에 따라 0차원은 scalar, 1차원은 벡터,
2차원은 행렬, 3차원은 3차원 행렬(텐서)라고
부른다.

>4차원 텐서의 경우 4차원 벡터처럼 수식으로만
표현되며, 보통 3차원 텐서(또는 이미지)가 여러 개
모여 있다는 의미가 된다.

### 합성곱 계산

![alt text](<../../../assets/img/ARM/AI/CNN/image copy 2.png>)

>차원의 크기가 같은 두 텐서를 계산해 Scalar 값이 되는 연산

### Filter 
![alt text](<../../../assets/img/ARM/AI/CNN/image copy 3.png>)
> 특정 사이즈의 텐서를 사용해 전체 이미지를 스캔하듯이 이동연산

> 이때 스캔하는 텐서를 필터라 한다.

### Feature Map
![alt text](<../../../assets/img/ARM/AI/CNN/image copy 4.png>)

- 입력 데이터에 필터로 스캔한 결과로 만들어지는 출력 텐서를 feature map(특성
맵)이라 한다.

- 보통 하나의 합성곱 층에서 여러 개의 필터가 사용되며, 그 결과로 필터 수만큼의 특성 맵이 만들어진다.

- 합성곱 연산 후에 활성화 함수를 적용하는데 이것은 특성 맵이 만들어진 후에 작용한다.

- 특성 맵에 활성화 함수를 작용시켜 만들어진 결과를 activation map이라고도 한다.

- CNN의 다음 층에서는 계산된 activation map들을 모아 하나의 텐서로 만들어서 새로운 입력으로 사용한다. 

- 예를 들어 activation map의 크기가 28×28이고 작용한
필터의 개수가 5개였다면, 다음 입력 데이터는 28×28×5의 크기가 된다. 
- 여기서 activation map과 특성 맵은 같은 크기이다.

- 그림을 살펴보면, 입력 이미지가 32×32×3이고 하나의 필터를 통해 28×28×1 크기를 갖는 하나의 특성 맵이 만들어지며, 여기에 activation function(예: ReLU)이 작용해 같은 크기의 activation map이 만들어진다.

### Stride
![alt text](<../../../assets/img/ARM/AI/CNN/image copy 5.png>)
> CNN 과정에서 합성곱 연산에 필터가 움직이는 간격

> 합성곱이 2차원(3x3) 으로 정의 따라서 Stride도 2차원(m, n)으로 정의

> 한 번 합성곱 연산을 한 후에 우측으로 m만큼씩 이동해 입력 텐서의 끝까지
이동한 후, 아래로 n만큼씩 움직여서 맨 왼쪽부터 다시 스캔하는 방식이다.

### Padding

![alt text](<../../../assets/img/ARM/AI/CNN/image copy 6.png>)

> 합성곱 연산이 특성 맵은 기존 데이터 크기보다 작아진다.

>패딩을 덧대어 출력 크기를 입력 크기와 같게 만들며 빈도수를 동일하게 설정할 수 있다.

> 합성곱 층에서 인수 설정을 Padding = same으로 사용

### 활성화 (Activation)

- 합성곱 연산 후, 신경망처럼 활성화 함수(ReLU 등) 를 사용하여 비선형성을 부여함.
  
- 일반적으로 ReLU 또는 ReLU 변형 함수를 사용.
  
- 이러한 처리 과정은 기존 신경망과 동일함.
  
- 합성곱 층 개수는 설정에 따라 달라짐 (1개만 사용하거나 2~3개 사용할 수도 있음).
  
- 각 합성곱 층 뒤에 Pooling Layer를 붙이기도 함.
  
- 여러 유명한 신경망 구조(Architecture)를 참고하면 더 좋은 모델을 설계하는 데 도움됨. (예: VGG, ResNet 등)


### Polling Layer

Max
![alt text](<../../../assets/img/ARM/AI/CNN/image copy 7.png>)

Average
![alt text](<../../../assets/img/ARM/AI/CNN/image copy 8.png>)

Global Average
![alt text](<../../../assets/img/ARM/AI/CNN/image copy 9.png>)

>합성곱 연산으로 만들어진 하나의 특성 맵에서 평균값을 출력하는 Pooling.

>이전의 Pooling 계산보다 크기를 많이 줄이게 된다.

>GoogLeNet에서 FC Layer 직전에 Flatten 대신 사용함.