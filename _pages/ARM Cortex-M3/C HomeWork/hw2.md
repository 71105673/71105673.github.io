---
title: "2차원 배열의 리턴" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---

## 문제
```c
#include <stdio.h>

       func(void)
{
    static int a[3][4] = {1,2,3,4,5,6,7,8,9,10,11,12};
    return a;
}

void main(void)
{
    printf("%d\n",     func()        );
}
```

## 출력 예시
```
7
```

## 정답
```c
#include <stdio.h>

int(*func(void))[4]
{
	static int a[3][4] = {1,2,3,4,5,6,7,8,9,10,11,12};
	return a;
}

void main(void)
{
	printf("%d\n",func()[1][2]);
}
```

