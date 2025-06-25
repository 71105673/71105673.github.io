---
title: "구조체 주소의 함수 전달" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---

## 문제
```c
#include <stdio.h> 
 
struct math
{
  int id;
  char name[20];
  int score;
};

void cheat(struct math * test);

void main(void)
{
  struct math final={1, "Kim", 50};
  cheat(&final);
  printf("%d\n", final.score);
}


// 함수에서 score를 100으로 수정하는 코드를 작성하라
// 단, -> 연산자는 사용할 수 없다


void cheat(struct math * test)
{
      = 100;
}
```

## 출력 설명
```
50이 100으로 수정되어 인쇄
```

## 출력 예시
```
100
```

## 정답
```c
#include <stdio.h>

struct math
{
	int id;
	char name[20];
	int score;
};

void cheat(struct math * test);

void main(void)
{
	struct math final = { 1, "Kim", 50 };
	cheat(&final);

	final.score = 100;

	printf("%d\n", final.score);  // 결과: 100
}

void cheat(struct math * test)
{
	test->score = 100;  // ← 여기가 핵심
}
```