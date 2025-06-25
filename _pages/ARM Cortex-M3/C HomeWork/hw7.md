---
title: "Type casting 연습 2" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---


## 문제
```c
 #include <stdio.h>
 
void func(void *p)
{
      printf("%s\n",          );
}
 
void main(void)
{
      char * p = "Willtek";
      func(&p);
}
```


## 정답
```c
#include <stdio.h>

void func(void *p)
{
	printf("%s\n", *(char **)p);
}

void main(void)
{
	char * p = "Willtek";

// 	printf("%s\n", p);
// 	printf("%s\n", *&p);

	func(&p);
}
```

