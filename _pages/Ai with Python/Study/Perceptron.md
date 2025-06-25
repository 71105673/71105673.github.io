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
print("\nOR Gate Test:"import numpy as np
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
# 퍼셉트론 모델 훈련
ppn_and = Perceptron(input_size=2)
ppn_and.train(X_and, y_and)

# 예측 결과 확인
print("\nAND Gate Test:")
for x in X_and:
    print(f"Input: {x}, Predicted Output: {ppn_and.predict(x)}"))
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

### 고찰 -> XOR의 Error

>선형 분리 불가능(linearly inseparable)

>XOR의 입력 값은 [0,0],[0,1],[1,0],[1,1]
>
>XOR의 출력 값은 [0, 1, 1, 0] 에 해당됩니다

>y=1 class는 [0,1], [1,0]
>
>y=0 class는 [0,0], [1,1]

>따라서 클래스 0과 클래스 1은 X자로 교차된다.
>
>따라서 한 개의 직선으로는 이 둘을 나눌 수 없다.

>결국 선형 결정 경계로 만드는 퍼셉트론은 하나의 직선으로 분류할 수 없기에 오류가 난 것이다.

>이를 해결하기 위해서는 비선형 결정 경계가 필요하다. 

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
mlp = MLPClassifier(hidden_layer_sizes=(4,),   # 은닉층 1개, 노드 4개
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

### 학습 로그
```
학습 로그
Epoch 1/117, Loss: 0.8091
Epoch 2/117, Loss: 0.7159
Epoch 3/117, Loss: 0.6959
Epoch 4/117, Loss: 0.7019
Epoch 5/117, Loss: 0.6989
Epoch 6/117, Loss: 0.6821
Epoch 7/117, Loss: 0.6606
Epoch 8/117, Loss: 0.6430
Epoch 9/117, Loss: 0.6324
Epoch 10/117, Loss: 0.6254
Epoch 11/117, Loss: 0.6160
Epoch 12/117, Loss: 0.6006
Epoch 13/117, Loss: 0.5795
Epoch 14/117, Loss: 0.5553
Epoch 15/117, Loss: 0.5312
Epoch 16/117, Loss: 0.5086
Epoch 17/117, Loss: 0.4866
Epoch 18/117, Loss: 0.4630
Epoch 19/117, Loss: 0.4368
Epoch 20/117, Loss: 0.4083
Epoch 21/117, Loss: 0.3786
Epoch 22/117, Loss: 0.3493
Epoch 23/117, Loss: 0.3213
Epoch 24/117, Loss: 0.2950
Epoch 25/117, Loss: 0.2702
Epoch 26/117, Loss: 0.2464
Epoch 27/117, Loss: 0.2235
Epoch 28/117, Loss: 0.2019
Epoch 29/117, Loss: 0.1818
Epoch 30/117, Loss: 0.1635
Epoch 31/117, Loss: 0.1472
Epoch 32/117, Loss: 0.1328
Epoch 33/117, Loss: 0.1200
Epoch 34/117, Loss: 0.1086
Epoch 35/117, Loss: 0.0983
Epoch 36/117, Loss: 0.0891
Epoch 37/117, Loss: 0.0808
Epoch 38/117, Loss: 0.0734
Epoch 39/117, Loss: 0.0667
Epoch 40/117, Loss: 0.0609
Epoch 41/117, Loss: 0.0558
Epoch 42/117, Loss: 0.0513
Epoch 43/117, Loss: 0.0474
Epoch 44/117, Loss: 0.0440
Epoch 45/117, Loss: 0.0411
Epoch 46/117, Loss: 0.0384
Epoch 47/117, Loss: 0.0361
Epoch 48/117, Loss: 0.0340
Epoch 49/117, Loss: 0.0322
Epoch 50/117, Loss: 0.0305
Epoch 51/117, Loss: 0.0290
Epoch 52/117, Loss: 0.0276
Epoch 53/117, Loss: 0.0263
Epoch 54/117, Loss: 0.0252
Epoch 55/117, Loss: 0.0241
Epoch 56/117, Loss: 0.0231
Epoch 57/117, Loss: 0.0222
Epoch 58/117, Loss: 0.0214
Epoch 59/117, Loss: 0.0206
Epoch 60/117, Loss: 0.0198
Epoch 61/117, Loss: 0.0191
Epoch 62/117, Loss: 0.0185
Epoch 63/117, Loss: 0.0178
Epoch 64/117, Loss: 0.0172
Epoch 65/117, Loss: 0.0167
Epoch 66/117, Loss: 0.0162
Epoch 67/117, Loss: 0.0157
Epoch 68/117, Loss: 0.0152
Epoch 69/117, Loss: 0.0147
Epoch 70/117, Loss: 0.0143
Epoch 71/117, Loss: 0.0139
Epoch 72/117, Loss: 0.0136
Epoch 73/117, Loss: 0.0132
Epoch 74/117, Loss: 0.0129
Epoch 75/117, Loss: 0.0126
Epoch 76/117, Loss: 0.0122
Epoch 77/117, Loss: 0.0120
Epoch 78/117, Loss: 0.0117
Epoch 79/117, Loss: 0.0114
Epoch 80/117, Loss: 0.0111
Epoch 81/117, Loss: 0.0109
Epoch 82/117, Loss: 0.0106
Epoch 83/117, Loss: 0.0104
Epoch 84/117, Loss: 0.0102
Epoch 85/117, Loss: 0.0100
Epoch 86/117, Loss: 0.0098
Epoch 87/117, Loss: 0.0096
Epoch 88/117, Loss: 0.0094
Epoch 89/117, Loss: 0.0092
Epoch 90/117, Loss: 0.0090
Epoch 91/117, Loss: 0.0089
Epoch 92/117, Loss: 0.0087
Epoch 93/117, Loss: 0.0086
Epoch 94/117, Loss: 0.0084
Epoch 95/117, Loss: 0.0083
Epoch 96/117, Loss: 0.0081
Epoch 97/117, Loss: 0.0080
Epoch 98/117, Loss: 0.0078
Epoch 99/117, Loss: 0.0077
Epoch 100/117, Loss: 0.0076
Epoch 101/117, Loss: 0.0075
Epoch 102/117, Loss: 0.0074
Epoch 103/117, Loss: 0.0072
Epoch 104/117, Loss: 0.0071
Epoch 105/117, Loss: 0.0070
Epoch 106/117, Loss: 0.0069
Epoch 107/117, Loss: 0.0068
Epoch 108/117, Loss: 0.0067
Epoch 109/117, Loss: 0.0066
Epoch 110/117, Loss: 0.0066
Epoch 111/117, Loss: 0.0065
Epoch 112/117, Loss: 0.0064
Epoch 113/117, Loss: 0.0063
Epoch 114/117, Loss: 0.0062
Epoch 115/117, Loss: 0.0062
Epoch 116/117, Loss: 0.0061
Epoch 117/117, Loss: 0.0060
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
![alt text](<../../../assets/img/ARM/AI/image copy 22.png>)

### 오류 시각화
```python
# 8. 손실 곡선 시각화
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("MLP Loss Curve (XOR Gate)")
plt.grid(True)
plt.show()
```
### 오류 시각화 결과
![alt text](<../../../assets/img/ARM/AI/image copy 23.png>)

---