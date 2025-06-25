---
title: "함수 등가포인터의 실행" 
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

void f1(void)
{
   printf("func\n");
}

int * f2(void)
{
   static int a[4] = {1,2,3,4};

   return a;
}

void main(void)
{
   // p, q, r 선언


   // p, q, r에 대응 함수 대입


   printf("%d\n", add(3,4));
   f1();
   printf("%d\n", f2()[2]);

   // 위와 동일한 결과가 나오도록 p, q, r로 실행
}
```

## 출력 예시
```
7
func
3
7
func
3
```

## 정답
```c
#include <stdio.h>

int add(int a, int b)
{
	return a+b;
}	

void f1(void)
{
	printf("func\n");
}

int * f2(void)
{
	static int a[4] = {1,2,3,4};

	return a;
}

void main(void)
{
	int(*p)(int, int);    // int형 두 개 받고 int형 반환하는 함수 포인터
	void(*q)(void);       // 인자 없고 반환 void인 함수 포인터
	int *(*r)(void);       // 인자 없고 int 포인터 반환하는 함수 포인터

	p = add;
	q = f1;
	r = f2;

	printf("%d\n", p(3, 4));
	q();
	printf("%d\n", r()[2]);

	printf("%d\n", add(3,4));
	f1();
	printf("%d\n", f2()[2]);
}
```

