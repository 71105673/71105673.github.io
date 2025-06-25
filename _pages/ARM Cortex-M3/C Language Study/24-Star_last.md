---
title: "별 자판기 - 트리별" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---

# 문제 설명
출력 예시 모양의 별을 인쇄하는 코드를 이중 for루프로 구현하라

## 문제

```c
#include <stdio.h>

void main(void)
{
	// 코드 작성
}
```
## 출력 예시
```
    *
   ***
  *****
 *******
*********
```

## 정답
```c
#include <stdio.h>

void main(void){
	int i, j;

	for (i = 1; i <= 5; i++) {
		for (j = 1; j <= 5 - i; j++) {
			printf(" ");
		}
		// 별 출력: 두 번째 루프에서는 2*i - 1 개의 *을 출력
		for (j = 1; j <= 2 * i - 1; j++) {
			printf("*");
		}

			printf("\n");
	}
	
}
```

