---
title: "함수를 함수에 전달하자" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---

## 문제
```c
#include <stdio.h> 

int add(int a, int b)
{
    return a+b;
}

int sub(int a, int b)
{
    return a-b;
}

void func(                  )
{
    printf("%d\n", p(3,4));
}

void main(void)
{
    func(add);
    func(sub);
}
```

## 출력 예시
```
7
-1
```

## 정답
```c
#include <stdio.h> 

int add(int a, int b)
{
    return a + b;
}

int sub(int a, int b)
{
    return a - b;
}

// 함수 포인터를 매개변수로 받는 func 함수
void func(int (*p)(int, int))
{
    printf("%d\n", p(3, 4));  // p가 가리키는 함수(add 또는 sub)를 호출
}

int main(void)
{
    func(add);
    func(sub);
}
```

