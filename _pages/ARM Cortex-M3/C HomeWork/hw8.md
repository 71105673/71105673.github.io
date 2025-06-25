---
title: " int 변수로 함수 실행하기" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---

## 문제
```c
 #include <stdio.h>
 
int func(int a, int b)
{
      return a+b;
}
     
void main(void)
{
      int a = (int)func;
 
      printf("%d\n", func(3,4));
      printf("%d\n",                 );
}
```


## 정답
```c
#include <stdio.h>

int func(int a, int b)
{
	return a+b;
}

typedef int(*FP)(int, int);
	
void main(void)
{
	int a = (int)func;

	printf("%d\n", func(3,4));
	printf("%d\n", ((FP)a)(3, 4));
	printf("%d\n", ((int (*)(int, int))a)(3, 4));
}
```

