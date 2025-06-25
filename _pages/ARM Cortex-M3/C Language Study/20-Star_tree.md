---
title: "별 자판기 - 삼각별" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---

# 출력 예시 모양의 별을 인쇄하는 코드를 이중 for루프로 구현하라

**문제**

```c
#include <stdio.h>

void main(void)
{
	// 코드 작성
}
```
```
출력 예시
*
**
***
****
*****
```

**정답**
```c
#include <stdio.h>

void main(void)
{
	int i, j;
	for (i = 0; i < 5; i++) {
		for (j = 0; j <= i; j++) {
			printf("*");
		}
		printf("\n");
	}
}
```

