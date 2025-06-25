---
title: " /, % 연산자의 활용 => 10진수 자리수 분리" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---

# 문제 설명
4자리 정수의 각 자리 값을 추출하는 다음 코드를 완성하라
## 문제
```c
#include <stdio.h>

void main(void)
{
	int a = 2345;
	int a4, a3, a2, a1;

	a4 = 
	a3 = 
	a2 = 
	a1 = 

	printf("1000자리=%d, 100자리=%d, 10자리=%d, 1자리=%d\n", a4, a3, a2, a1);
}
```


## 출력 예시
```
1000자리=2, 100자리=3, 10자리=4, 1자리=5
```

## 정답
```c
#include <stdio.h>

void main(void)
{
	int a = 2345;
	int a4, a3, a2, a1;

	a4 = a / 1000;
	a3 = (a / 100) % 10;
	a2 = (a / 10) % 10;
	a1 = a % 10;

		printf("1000자리=%d, 100자리=%d, 10자리=%d, 1자리=%d\n", a4, a3, a2, a1);
}
```

