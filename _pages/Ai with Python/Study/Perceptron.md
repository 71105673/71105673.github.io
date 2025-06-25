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
>epochs -> 학습 횟수

>lr -> learning rate (학습률) 
>
>lr = 0.1 → 보통 시작값으로 적절
>
>lr = 0.01 → 더 느리지만 안정적
>
>lr = 1.0 → 너무 크면 발산할 수 있음

### 학습 로그
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
```
### 예측 결과
```
AND Gate Test:
Input: [0 0], Predicted Output:과 0
Input: [0 1], Predicted Output: 0
Input: [1 0], Predicted Output: 0
Input: [1 1], Predicted Output: 1
```

### 경계 결정 시각화
```python
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    h = .02  # mesh grid 간격
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # 실제 데이터 포인트 표시
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=100, marker='o')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Perceptron Decision Boundary')
    plt.show()

# AND 게이트 결정 경계 시각화
plot_decision_boundary(X_and, y_and, ppn_and)
```
### 경계 결정 시각화 결과
![alt text](<../../../assets/img/ARM/AI/image copy 11.png>)

### 오류 시각화
```python
#오류 시각화
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ppn_and.errors) + 1), ppn_and.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.title('Perceptron Learning Error Over Epochs (And Gate)')
plt.grid(True)
plt.show()
```
### 오류 시각화 결과
![alt text](<../../../assets/img/ARM/AI/image copy 12.png>)

---
**2. OR**
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
      
# OR 게이트 데이터
X_or = np.array([[0,0],[0,1],[1,0],[1,1]])
y_or = np.array([0,1,1,1])

# 퍼셉트론 모델 훈련
ppn_or = Perceptron(input_size=2)
ppn_or.train(X_or, y_or)

# 예측 결과 확인
print("\nOR Gate Test:")
for x in X_or:
    print(f"Input: {x}, Predicted Output: {ppn_or.predict(x)}")

```
### 학습 로그
```
Epoch 1/10, Errors: 1
Epoch 2/10, Errors: 2
Epoch 3/10, Errors: 1
Epoch 4/10, Errors: 0
Epoch 5/10, Errors: 0
Epoch 6/10, Errors: 0
Epoch 7/10, Errors: 0
Epoch 8/10, Errors: 0
Epoch 9/10, Errors: 0
Epoch 10/10, Errors: 0
```
### 예측 결과
```
OR Gate Test:
Input: [0 0], Predicted Output: 0
Input: [0 1], Predicted Output: 1
Input: [1 0], Predicted Output: 1
Input: [1 1], Predicted Output: 1
```
### 경계 결정 시각화
```python
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    h = .02  # mesh grid 간격
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # 실제 데이터 포인트 표시
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=100, marker='o')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Perceptron Decision Boundary')
    plt.show()

# OR 게이트 결정 경계 시각화
plot_decision_boundary(X_or, y_or, ppn_or)
```

### 경계 결정 시각화 결과

![alt text](<../../../assets/img/ARM/AI/image copy 13.png>)

### 오류 시각화
```python
#오류 시각화
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ppn_or.errors) + 1), ppn_or.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.title('Perceptron Learning Error Over Epochs (OR Gate)')
plt.grid(True)
plt.show()
```

### 오류 시각화 결과
![alt text](<../../../assets/img/ARM/AI/image copy 18.png>)

---
**3. NAND**
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
      
# OR 게이트 데이터
X_nand = np.array([[0,0],[0,1],[1,0],[1,1]])
y_nand = np.array([1,1,1,0])

# 퍼셉트론 모델 훈련
ppn_nand = Perceptron(input_size=2)
ppn_nand.train(X_nand, y_nand)

# 예측 결과 확인
print("\nNAND Gate Test:")
for x in X_nand:
    print(f"Input: {x}, Predicted Output: {ppn_nand.predict(x)}")
```
### 학습 로그
```
Epoch 1/10, Errors: 2
Epoch 2/10, Errors: 3
Epoch 3/10, Errors: 3
Epoch 4/10, Errors: 0
Epoch 5/10, Errors: 0
Epoch 6/10, Errors: 0
Epoch 7/10, Errors: 0
Epoch 8/10, Errors: 0
Epoch 9/10, Errors: 0
Epoch 10/10, Errors: 0
```
### 예측 결과
```
NAND Gate Test:
Input: [0 0], Predicted Output: 1
Input: [0 1], Predicted Output: 1
Input: [1 0], Predicted Output: 1
Input: [1 1], Predicted Output: 0

```
### 경계 결정 시각화
```python
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    h = .02  # mesh grid 간격
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # 실제 데이터 포인트 표시
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=100, marker='o')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Perceptron Decision Boundary')
    plt.show()

# NAND 게이트 결정 경계 시각화
plot_decision_boundary(X_nand, y_nand, ppn_nand)
```

### 경계 결정 시각화 결과
![alt text](<../../../assets/img/ARM/AI/image copy 15.png>)

### 오류 시각화
```python
#오류 시각화
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ppn_nand.errors) + 1), ppn_nand.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.title('Perceptron Learning Error Over Epochs (NAND Gate)')
plt.grid(True)
plt.show()
```
### 오류 시각화 결과
![alt text](<../../../assets/img/ARM/AI/image copy 17.png>)
---

**4. XOR**
```python
```
### 학습 로그
```
```
### 예측 결과
```
```
### 경계 결정 시각화
```python
```

### 경계 결정 시각화 결과

### 오류 시각화
```python
```

### 오류 시각화 결과
---