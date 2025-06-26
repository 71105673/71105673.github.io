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

## 2-1. 구조
```
입력(x) → 가중치(w) → 가중합(Σ) → 활성화 함수(f) → 출력(y)
```
![alt text](<../../../assets/img/ARM/AI/image copy 26.png>)
- 입력(Input) : AND 또는 OR 연산을 위한 입력 신호

- 가중치(Weight) : 입력 신호에 부여되는 중요도로, 가중치가 크다는 것은 그 입력이 출력을 결정하는 데 큰 역할을 한다는 의미

- 가중합(Weighted Sum) : 입력값과 가중치의 곱을 모두 합한 값

![alt text](<../../../assets/img/ARM/AI/image copy 25.png>)

- 활성화 함수(Activation Function) : 어떠한 신호를 입력받아 이를 적절하게 처리하여 출력해 주는 함수로, 가중합이 임계치(Threshold)를 넘으면 1, 그렇지 않으면 0을 출력함
  
- 출력(Output) : 최종 결과(분류)
---

## 2.-. 동작 방식

**Perceptron의 작동 방식**

퍼셉트론은 지도 학습(supervised learning) 방식으로 학습한다. 즉, 입력 데이터와 그에 해당하는 정답(레이블)을 함께 사용하여 모델을 훈련시키며 학습 과정은 다음과 같다.

1. 초기화: 가중치(wi)와 편향(b)을 작은 임의의 값으로 초기화한다.

2. 예측: 훈련 데이터의 한 샘플을 입력으로 받아 가중합을 계산하고, 활성화 함수를 적용하여 출력 값을 예측한다.

3. 오류 계산: 예측된 출력 값과 실제 정답(레이블)을 비교하여 오류를 계산한다.

4. 가중치 및 편향 업데이트: 오류가 발생하면, 오류를 줄이는 방향으로 가중치와 편향을 업데이트하며 이 업데이트는 일반적으로 다음 규칙을 따른다.

    ![alt text](<../../../assets/img/ARM/AI/image copy 24.png>)

    여기서 η (eta)는 학습률(learning rate)로, 가중치를 얼마나 크게 업데이트할지를 결정하는 값이다. y는 실제 정답, y^는 예측 값을 의미한다.

5. 반복: 모든 훈련 데이터 샘플에 대해 2~4단계를 반복한다. 이 과정을 여러 번 반복하여 (에포크, epoch) 가중치와 편향이 최적의 값에 수렴하도록 한다.

>epochs -> 학습 횟수

>lr -> learning rate (학습률) 
>
>lr = 0.1 → 보통 시작값으로 적절
>
>lr = 0.01 → 더 느리지만 안정적
>
>lr = 1.0 → 너무 크면 발산할 수 있음

---

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
      
# NAND 게이트 데이터
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

# XOR 게이트 데이터
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

# 퍼셉트론 모델 훈련
ppn_xor = Perceptron(input_size=2)
ppn_xor.train(X_xor, y_xor)

# 예측 결과 확인
print("\nXOR Gate Test:")
for x in X_xor:
    print(f"Input: {x}, Predicted Output: {ppn_xor.predict(x)}")
```
### 학습 로그
```
Epoch 1/10, Errors: 2
Epoch 2/10, Errors: 3
Epoch 3/10, Errors: 4
Epoch 4/10, Errors: 4
Epoch 5/10, Errors: 4
Epoch 6/10, Errors: 4
Epoch 7/10, Errors: 4
Epoch 8/10, Errors: 4
Epoch 9/10, Errors: 4
Epoch 10/10, Errors: 4
```
### 예측 결과
```
XOR Gate Test:
Input: [0 0], Predicted Output: 1
Input: [0 1], Predicted Output: 1
Input: [1 0], Predicted Output: 0
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

# XOR 게이트 결정 경계 시각화
plot_decision_boundary(X_xor, y_xor, ppn_xor)
```

### 경계 결정 시각화 결과
![alt text](<../../../assets/img/ARM/AI/image copy 19.png>)

### 오류 시각화
```python
#오류 시각화
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ppn_xor.errors) + 1), ppn_xor.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.title('Perceptron Learning Error Over Epochs (XOR Gate)')
plt.grid(True)
plt.show()
```
### 오류 시각화 결과
![alt text](<../../../assets/img/ARM/AI/image copy 20.png>)

---

### 고찰 -> XOR의 Error 이유

> AND, OR, NAND gate 들은 선형 분리 가능. (Z=wx+b)​
>
>즉, 데이터 포인트를 두 그룹으로 나눌 수 있음. ​
>
>그러나, XOR의 경우 데이터를 두 그룹으로 분리가 불가능.​
>
>![alt text](<../../../assets/img/ARM/AI/image copy 27.png>)
>
>선형 분리 불가능(linearly inseparable)
>
>XOR의 입력 값은 [0,0],[0,1],[1,0],[1,1]
>
>XOR의 출력 값은 [0, 1, 1, 0] 에 해당됩니다
>
>y=1 class는 [0,1], [1,0]
>
>y=0 class는 [0,0], [1,1]
>
>따라서 클래스 0과 클래스 1은 X자로 교차된다.
>
>따라서 한 개의 직선으로는 이 둘을 나눌 수 없다.
>
>결국 선형 결정 경계로 만드는 퍼셉트론은 하나의 직선으로 분류할 수 없기에 오류가 난 것이다.
>
>이를 해결하기 위해서는 비선형 결정 경계가 필요하다. 

---

### MLP (Multi Layer Perceptron)​

>![alt text](<../../../assets/img/ARM/AI/스크린샷 2025-06-26 09-00-23.png>)
>
>hidden Layer를 통해 입출력 사이의 복잡한 패턴을 추출하고 학습한다.

>
>![alt text](<../../../assets/img/ARM/AI/스크린샷 2025-06-26 09-00-37.png>)
>
>순전파 : 입력 데이터에 가중합 계산을 적용하여 활성화 함수를 통과시켜 최종 출력을 생성.​
>
>역전파 : 출력에서 발생한 오차를 바탕으로 가중치와 편향을 갱신. (오차 최소화)​

>![alt text](<../../../assets/img/ARM/AI/스크린샷 2025-06-26 09-00-50.png>)
>
>비선형 이진 분류를 위한 활성화 함수 출력 값을 확률로 판단하여 정함.

### XOR_MLP -> 해결 방안

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from matplotlib.colors import ListedColormap

# 1. XOR 데이터
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# 2. MLP 모델 정의 (비선형 해결 가능)
mlp = MLPClassifier(hidden_layer_sizes=(2,),   # 은닉층 1개, 노드 2개
                    activation='tanh',         # 비선형 활성화 함수
                    solver='adam',
                    learning_rate_init=0.1,
                    max_iter=1000,
                    random_state=42)

# 3. 훈련
mlp.fit(X, y)

# 4. 학습 로그 출력
print("학습 로그")
for i, loss in enumerate(mlp.loss_curve_):
    print(f"Epoch {i+1}/{len(mlp.loss_curve_)}, Loss: {loss:.4f}")

# 5. 예측 결과 출력
print("\n예측 결과")
print("XOR Gate Test:")
for x in X:
    pred = mlp.predict([x])[0]
    print(f"Input: {x}, Predicted Output: {pred}")
```

### 학습 로그 (노드 2개)
```
학습 로그
Epoch 1/202, Loss: 0.8554
Epoch 2/202, Loss: 0.7697
Epoch 3/202, Loss: 0.7273
Epoch 4/202, Loss: 0.7169
Epoch 5/202, Loss: 0.7138
Epoch 6/202, Loss: 0.7074
Epoch 7/202, Loss: 0.6985
Epoch 8/202, Loss: 0.6910
Epoch 9/202, Loss: 0.6874
Epoch 10/202, Loss: 0.6880
Epoch 11/202, Loss: 0.6903
Epoch 12/202, Loss: 0.6921
Epoch 13/202, Loss: 0.6921
Epoch 14/202, Loss: 0.6908
Epoch 15/202, Loss: 0.6889
Epoch 16/202, Loss: 0.6874
Epoch 17/202, Loss: 0.6867
Epoch 18/202, Loss: 0.6867
Epoch 19/202, Loss: 0.6868
Epoch 20/202, Loss: 0.6865
Epoch 21/202, Loss: 0.6853
Epoch 22/202, Loss: 0.6829
Epoch 23/202, Loss: 0.6795
Epoch 24/202, Loss: 0.6755
Epoch 25/202, Loss: 0.6708
Epoch 26/202, Loss: 0.6656
Epoch 27/202, Loss: 0.6596
Epoch 28/202, Loss: 0.6525
Epoch 29/202, Loss: 0.6446
Epoch 30/202, Loss: 0.6360
Epoch 31/202, Loss: 0.6272
Epoch 32/202, Loss: 0.6185
Epoch 33/202, Loss: 0.6096
Epoch 34/202, Loss: 0.6004
Epoch 35/202, Loss: 0.5907
Epoch 36/202, Loss: 0.5808
Epoch 37/202, Loss: 0.5706
Epoch 38/202, Loss: 0.5601
Epoch 39/202, Loss: 0.5490
Epoch 40/202, Loss: 0.5374
Epoch 41/202, Loss: 0.5255
Epoch 42/202, Loss: 0.5139
Epoch 43/202, Loss: 0.5030
Epoch 44/202, Loss: 0.4932
Epoch 45/202, Loss: 0.4841
Epoch 46/202, Loss: 0.4753
Epoch 47/202, Loss: 0.4664
Epoch 48/202, Loss: 0.4573
Epoch 49/202, Loss: 0.4485
Epoch 50/202, Loss: 0.4405
Epoch 51/202, Loss: 0.4337
Epoch 52/202, Loss: 0.4277
Epoch 53/202, Loss: 0.4222
Epoch 54/202, Loss: 0.4168
Epoch 55/202, Loss: 0.4115
Epoch 56/202, Loss: 0.4066
Epoch 57/202, Loss: 0.4023
Epoch 58/202, Loss: 0.3985
Epoch 59/202, Loss: 0.3952
Epoch 60/202, Loss: 0.3922
Epoch 61/202, Loss: 0.3894
Epoch 62/202, Loss: 0.3868
Epoch 63/202, Loss: 0.3844
Epoch 64/202, Loss: 0.3823
Epoch 65/202, Loss: 0.3804
Epoch 66/202, Loss: 0.3787
Epoch 67/202, Loss: 0.3771
Epoch 68/202, Loss: 0.3757
Epoch 69/202, Loss: 0.3743
Epoch 70/202, Loss: 0.3730
Epoch 71/202, Loss: 0.3718
Epoch 72/202, Loss: 0.3707
Epoch 73/202, Loss: 0.3697
Epoch 74/202, Loss: 0.3689
Epoch 75/202, Loss: 0.3681
Epoch 76/202, Loss: 0.3673
Epoch 77/202, Loss: 0.3665
Epoch 78/202, Loss: 0.3658
Epoch 79/202, Loss: 0.3652
Epoch 80/202, Loss: 0.3645
Epoch 81/202, Loss: 0.3640
Epoch 82/202, Loss: 0.3635
Epoch 83/202, Loss: 0.3630
Epoch 84/202, Loss: 0.3626
Epoch 85/202, Loss: 0.3621
Epoch 86/202, Loss: 0.3617
Epoch 87/202, Loss: 0.3613
Epoch 88/202, Loss: 0.3609
Epoch 89/202, Loss: 0.3606
Epoch 90/202, Loss: 0.3602
Epoch 91/202, Loss: 0.3599
Epoch 92/202, Loss: 0.3596
Epoch 93/202, Loss: 0.3593
Epoch 94/202, Loss: 0.3590
Epoch 95/202, Loss: 0.3588
Epoch 96/202, Loss: 0.3585
Epoch 97/202, Loss: 0.3582
Epoch 98/202, Loss: 0.3580
Epoch 99/202, Loss: 0.3578
Epoch 100/202, Loss: 0.3575
Epoch 101/202, Loss: 0.3573
Epoch 102/202, Loss: 0.3571
Epoch 103/202, Loss: 0.3569
Epoch 104/202, Loss: 0.3567
Epoch 105/202, Loss: 0.3564
Epoch 106/202, Loss: 0.3562
Epoch 107/202, Loss: 0.3560
Epoch 108/202, Loss: 0.3558
Epoch 109/202, Loss: 0.3556
Epoch 110/202, Loss: 0.3554
Epoch 111/202, Loss: 0.3552
Epoch 112/202, Loss: 0.3550
Epoch 113/202, Loss: 0.3547
Epoch 114/202, Loss: 0.3545
Epoch 115/202, Loss: 0.3543
Epoch 116/202, Loss: 0.3540
Epoch 117/202, Loss: 0.3538
Epoch 118/202, Loss: 0.3535
Epoch 119/202, Loss: 0.3532
Epoch 120/202, Loss: 0.3529
Epoch 121/202, Loss: 0.3526
Epoch 122/202, Loss: 0.3522
Epoch 123/202, Loss: 0.3518
Epoch 124/202, Loss: 0.3514
Epoch 125/202, Loss: 0.3509
Epoch 126/202, Loss: 0.3503
Epoch 127/202, Loss: 0.3497
Epoch 128/202, Loss: 0.3490
Epoch 129/202, Loss: 0.3481
Epoch 130/202, Loss: 0.3471
Epoch 131/202, Loss: 0.3459
Epoch 132/202, Loss: 0.3445
Epoch 133/202, Loss: 0.3427
Epoch 134/202, Loss: 0.3404
Epoch 135/202, Loss: 0.3374
Epoch 136/202, Loss: 0.3336
Epoch 137/202, Loss: 0.3284
Epoch 138/202, Loss: 0.3212
Epoch 139/202, Loss: 0.3113
Epoch 140/202, Loss: 0.2973
Epoch 141/202, Loss: 0.2778
Epoch 142/202, Loss: 0.2516
Epoch 143/202, Loss: 0.2190
Epoch 144/202, Loss: 0.1827
Epoch 145/202, Loss: 0.1477
Epoch 146/202, Loss: 0.1179
Epoch 147/202, Loss: 0.0938
Epoch 148/202, Loss: 0.0750
Epoch 149/202, Loss: 0.0608
Epoch 150/202, Loss: 0.0507
Epoch 151/202, Loss: 0.0439
Epoch 152/202, Loss: 0.0394
Epoch 153/202, Loss: 0.0365
Epoch 154/202, Loss: 0.0344
Epoch 155/202, Loss: 0.0327
Epoch 156/202, Loss: 0.0312
Epoch 157/202, Loss: 0.0296
Epoch 158/202, Loss: 0.0278
Epoch 159/202, Loss: 0.0258
Epoch 160/202, Loss: 0.0239
Epoch 161/202, Loss: 0.0220
Epoch 162/202, Loss: 0.0203
Epoch 163/202, Loss: 0.0188
Epoch 164/202, Loss: 0.0174
Epoch 165/202, Loss: 0.0163
Epoch 166/202, Loss: 0.0153
Epoch 167/202, Loss: 0.0145
Epoch 168/202, Loss: 0.0138
Epoch 169/202, Loss: 0.0131
Epoch 170/202, Loss: 0.0126
Epoch 171/202, Loss: 0.0122
Epoch 172/202, Loss: 0.0118
Epoch 173/202, Loss: 0.0114
Epoch 174/202, Loss: 0.0111
Epoch 175/202, Loss: 0.0108
Epoch 176/202, Loss: 0.0106
Epoch 177/202, Loss: 0.0104
Epoch 178/202, Loss: 0.0101
Epoch 179/202, Loss: 0.0100
Epoch 180/202, Loss: 0.0098
Epoch 181/202, Loss: 0.0096
Epoch 182/202, Loss: 0.0094
Epoch 183/202, Loss: 0.0093
Epoch 184/202, Loss: 0.0091
Epoch 185/202, Loss: 0.0090
Epoch 186/202, Loss: 0.0088
Epoch 187/202, Loss: 0.0087
Epoch 188/202, Loss: 0.0086
Epoch 189/202, Loss: 0.0085
Epoch 190/202, Loss: 0.0084
Epoch 191/202, Loss: 0.0083
Epoch 192/202, Loss: 0.0082
Epoch 193/202, Loss: 0.0081
Epoch 194/202, Loss: 0.0080
Epoch 195/202, Loss: 0.0079
Epoch 196/202, Loss: 0.0078
Epoch 197/202, Loss: 0.0077
Epoch 198/202, Loss: 0.0077
Epoch 199/202, Loss: 0.0076
Epoch 200/202, Loss: 0.0075
Epoch 201/202, Loss: 0.0074
Epoch 202/202, Loss: 0.0074
```
### 예측 결과
```
예측 결과
XOR Gate Test:
Input: [0 0], Predicted Output: 0
Input: [0 1], Predicted Output: 1
Input: [1 0], Predicted Output: 1
Input: [1 1], Predicted Output: 0
```
### 경계 결정 시각화
```python
# 6. 결정 경계 시각화 함수
def plot_decision_boundary_proba(X, y, model):
    cmap_light = ListedColormap(['#FFBBBB', '#BBBBFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    h = .01
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z_proba.reshape(xx.shape)

    # 시각화
    plt.figure(figsize=(8, 6))
    cs = plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 100), cmap=cmap_light, alpha=0.9)
    plt.colorbar(cs, label='Probability of Class 1')

    # 결정 경계 (0.5 기준)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    # 데이터 점 표시
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=100)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.title("MLP Decision Boundary (XOR Gate)")
    plt.grid(True)
    plt.show()

# 7. 결정 경계 시각화
plot_decision_boundary_proba(X, y, mlp)
```
### 경계 결정 시각화 결과
> 노드 2개
>![alt text](<../../../assets/img/ARM/AI/image copy 29.png>)

> 노드 4개
>![alt text](<../../../assets/img/ARM/AI/image copy 22.png>)

### 오류 시각화
```python
# 8. 손실 곡선 시각화
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("MLP Loss Curve (XOR Gate)")
plt.grid(True)
plt.show()개
```
### 오류 시각화 결과
> 노드 2개
>![alt text](<../../../assets/img/ARM/AI/image copy 28.png>)

> 노드 4개
>![alt text](<../../../assets/img/ARM/AI/image copy 23.png>)
---