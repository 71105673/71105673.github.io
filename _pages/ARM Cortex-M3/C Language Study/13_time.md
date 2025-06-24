---
title: "시간차 구하기" 
date: "2025-03-25"
thumbnail: "../../../assets/img/C_image.png"
---

#  HH:MM:SS(시:분:초)의 형태로 표시하는 2개의 시간이 주어졌을 때, 시간차를 구하는 프로그램을 작성한다. 

**2개의 시간은 최대 24시간 차이가 난다고 가정한다.**
---

**문제**

```c
#include <stdio.h>

int main(void)
{
	int h1, m1, s1, h2, m2, s2;
	int h, m, s;

	// 입력받는 부분
	scanf("%d:%d:%d", &h1, &m1, &s1);
	scanf("%d:%d:%d", &h2, &m2, &s2);

	// 여기서부터 작성


	// 출력하는 부분
	printf("%02d:%02d:%02d", h, m, s);

	return 0;
}
```

```
입력 예시
20:00:00
04:00:00
```

```
출력 예시
08:00:00
```

**정답**
```c
#include <stdio.h>

int main(void)
{
	int h1, m1, s1, h2, m2, s2;
	int h, m, s;

	scanf("%d:%d:%d", &h1, &m1, &s1);
	scanf("%d:%d:%d", &h2, &m2, &s2);

	int time_1 = h1 * 3600 + m1 * 60 + s1;
	int time_2 = h2 * 3600 + m2 * 60 + s2;
	
	if (time_2 <= time_1) {
		time_2 += 24 * 3600; // 24시간을 더해줌
	}

	int k = time_2 - time_1;

	h = k / 3600;
	m = (k % 3600) / 60;
	s = k % 60;
	

	printf("%02d:%02d:%02d", h, m, s);


	return 0;
}
```

