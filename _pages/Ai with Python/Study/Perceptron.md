---
title: "Day-3 Perceptron"
date: "2025-06-25"
thumbnail: "../../../assets/img/ARM/AI/image copy 10.png"
---

## 1. Perceptron 이란?
- 인간의 뇌를 기계적으로 모델링
  
- 퍼셉트론은 생물학적 뉴런을 수학적으로 모델링한 '인공 뉴런'으로, 여러 입력 신호를 받아 각각의 가중치를 곱한 후
이를 합산해 활성화 함수를 통해 단일 신호를 출력한다.

- 퍼셉트론의 출력은 신호 유무(1 또는 0)로 표현되며, 이진 분류 문제 해결에 효과적이다.

- 입력 신호의 중요성을 나타내는 가중치는 머신러닝의 '학습' 과정에서 조정된다.

## 2. 구조
```
입력(x) → 가중치(w) → 가중합(Σ) → 활성화 함수(f) → 출력(y)
```
- 입력(Input) : AND 또는 OR 연산을 위한 입력 신호

- 가중치(Weight) : 입력 신호에 부여되는 중요도로, 가중치가 크다는 것은 그 입력이 출력을 결정하는 데 큰 역할을 한다는 의미

- 가중합(Weighted Sum) : 입력값과 가중치의 곱을 모두 합한 값

- 활성화 함수(Activation Function) : 어떠한 신호를 입력받아 이를 적절하게 처리하여 출력해 주는 함수로, 가중합이 임계치(Threshold)를 넘으면 1, 그렇지 않으면 0을 출력함

- 출력(Output) : 최종 결과(분류)

## 3. 실습 

**1. AND**

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = lr
        self.epochs = epochs
        self.errors = []

    def activation(self, x):
        return np.where(x > 0, 1, 0)

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return self.activation(linear_output)

    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                update = self.lr * (target - prediction)
                self.weights += update * xi
                self.bias += update
                total_error += int(update != 0.0)
            self.errors.append(total_error)
            print(f"Epoch {epoch+1}/{self.epochs}, Errors: {total_error}")
      
# AND 게이트 데이터
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])

# 퍼셉트론 모델 훈련
ppn_and = Perceptron(input_size=2)
ppn_and.train(X_and, y_and)

# 예측 결과 확인
print("\nAND Gate Test:")
for x in X_and:
    print(f"Input: {x}, Predicted Output: {ppn_and.predict(x)}")
```
### 결과
```
Epoch 1/10, Errors: 1
Epoch 2/10, Errors: 3
Epoch 3/10, Errors: 3
Epoch 4/10, Errors: 2
Epoch 5/10, Errors: 1
Epoch 6/10, Errors: 0
Epoch 7/10, Errors: 0
Epoch 8/10, Errors: 0
Epoch 9/10, Errors: 0
Epoch 10/10, Errors: 0

AND Gate Test:
Input: [0 0], Predicted Output: 0
Input: [0 1], Predicted Output: 0
Input: [1 0], Predicted Output: 0
Input: [1 1], Predicted Output: 1
```
 ---

**OR**

---

**NAND**

---

**XOR**

---