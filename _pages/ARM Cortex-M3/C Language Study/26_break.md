---
title: "j == 4 일 때 완전히 루프를 탈출" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---

# 문제 설명
j == 4일 때 완전히 바깥 루프까지 빠져 나오려면?

*break문은 가장 가까이에 있는 for loop를 빠져 나온다*


## 문제

```c
#include <stdio.h>

void main(void)
{
	int i, j;

	for (i = 0; i < 20; i++)
	{
		for (j = 0; j < 10; i++, j++)
		{
			if (j == 4) break;
			printf("%d %d\n", i, j);
		}

		if (i % 3) continue;
	}
}
```
## 출력 예시
```
0 0
1 1
2 2
3 3
```

## 정답
```c
#include <stdio.h>

void main(void)
{
	int i, j;
	int M;

	for (i = 0; i < 20; i++)
	{
		for (j = 0; j < 10; i++, j++)
		{
			if (j == 4) {
				M = 1;
				break;
			}
		printf("%d %d\n", i, j);
		}
		if (M != 0) {
			break;
		}
		if (i % 3) continue;
	}
}
```

