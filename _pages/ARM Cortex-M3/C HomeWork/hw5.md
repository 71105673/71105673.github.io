---
title: "함수 Lookup table (코드 수정)" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---

## 문제
```c
#include <stdio.h>
#include <stdlib.h>

int add(int a, int b)
{
    return a+b;
}

int sub(int a, int b)
{
    return a-b;
}

int mul(int a, int b)
{
    return a*b;
}

int get_key(void)
{
    static int r = 0;
    int ret = r;

    r = (r + 1) % 3;
    return ret;
}

       fa[3]       = {add, sub, mul};

int op(int a, int b)
{
    return fa[get_key()](a,b);
}

void main(void)
{
    printf("%d\n", op(3, 4));
    printf("%d\n", op(3, 4));
    printf("%d\n", op(3, 4));
    printf("%d\n", op(3, 4));
    printf("%d\n", op(3, 4));
    printf("%d\n", op(3, 4));
    printf("%d\n", op(3, 4));
}
```

## 출력 예시
```
7
-1
12
7
-1
12
7
```

## 정답
```c
#include <stdio.h> 
#include <stdlib.h> 

int add(int a, int b)
{
	return a + b;
}

int sub(int a, int b)
{
	return a - b;
}

int mul(int a, int b)
{
	return a * b;
}

int get_key(void)
{
	static int r = 0;
	int ret = r;

	r = (r + 1) % 3;  // 0, 1, 2 순서로 반복
	return ret;
}

// 함수 포인터 배열 선언
int(*fa[3])(int, int) = { add, sub, mul };

int op(int a, int b)
{
	return fa[get_key()](a, b);  
}

int main(void)
{
	printf("%d\n", op(3, 4));  
	printf("%d\n", op(3, 4));  
	printf("%d\n", op(3, 4));  
	printf("%d\n", op(3, 4));  
	printf("%d\n", op(3, 4));  
	printf("%d\n", op(3, 4));  
	printf("%d\n", op(3, 4));  
}
```

