---
title: "Type casting 연습 1 (오류 무시)" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---

## 문제
```c
 #include<stdio.h>
 
void func(void * p)
{
      int i;
 
      for(i=0; i<3; i++)
      {
            printf("%f\n",              );
      }
}
 
void main(void)
{
      double d[3] = {3.14, 5.125, -7.42};
      void *p = d;
 
      func(&p);
}
```

## 정답
```c
#include <stdio.h>

void func(void *p)
{
	int i;
	for (i = 0; i < 3; i++)
	{
		printf("%f\n", (*(double **)p)[i]);
	}
}

void main(void)
{
	double d[3] = { 3.14, 5.125, -7.42 };
	void *p = d;
	func(&p);  
}
```

