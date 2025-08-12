---
title: "System Verolog Day-5"
date: "2025-08-12"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# 0~10까지 전부 더하기
```
i = 0;
sum = 0;
while(i<=10) {
    sum = sum + i;
    i = i + 1;
    outport = sum;
}
halt;
```
## DataPath
![alt text](<../../../assets/img/CPU/day_5/스크린샷 2025-08-12 101431.png>)

## Control Unit 위한 State 및 Signal
![alt text](<../../../assets/img/CPU/day_5/스크린샷 2025-08-12 103030.png>)

이때, **i = i + 1 (S3 state)** 에서 **OutBuf가 1**이면, **Outport=sum(S4)** 를 줄이고 동작에 이상 없게 할 수 있다.

![alt text](../../../assets/img/CPU/day_5/KakaoTalk_20250812_104235890.jpg)

## 최종 버전
![alt text](<../../../assets/img/CPU/day_5/스크린샷 2025-08-12 104703.png>)