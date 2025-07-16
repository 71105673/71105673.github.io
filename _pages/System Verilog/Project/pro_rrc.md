---
title: "project RRC" 
date: "2025-07-16"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# 과제 기본 개념 정리
![text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-15 161918.png>) 
![text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-15 162201.png>) 
![text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-15 162632.png>) 
![text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-15 163210.png>)
- 고정소숫점 방식 -> 비트수가 작으면 노이즈가 증가, 그러나 area는 작음
 
- 이것을 fixed point simulation

![alt text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-15 163719.png>)
- RRC Filter -> 주파수 도메인 확인

- 여기서 fixed = 1.8인 이유가 정수부가 작아서

- 즉 정수부 1, 소수부 8 = 9bit

# 프로젝트 
> data = <1.6>, coeff = <1.8>
> 
> 즉, 7 x 9 = 16bit가 필요하다. / <2.14>

>**2Tap 기준 예시**
> <2.14> + <2.14> = <3.14> 
>
>**16Tap 이면??**
> <2.14> + <2.14> + ... = <6.14>
>
>**64Tap 이면??** 
> <2.14> + <2.14> + ... = <8.14>

만약 <1.6>을 <2.5> format으로 하면 Gain이 작아짐

마지막에 <8.14> => <1.6>으로 해야함

상위 7bit -> saturation

하위 7bit -> Truncation

## 🚫 Saturation (상위 비트 포화 처리)

- 정수부 8비트를 1비트로 줄이면 범위 초과가 발생할 수 있음
- **범위 초과 시**, 최대/최소값으로 고정

| 원래 값 (8.14 기준)         | 변환 결과 (1.6 기준)       | 설명                |
|-----------------------------|-----------------------------|---------------------|
| 00000001.xxxxxxxxxxxxxx     | 그대로 유지                | 표현 가능           |
| 01111111.xxxxxxxxxxxxxx     | 1.111111                    | 양의 최대값으로 포화 |
| 10000000.xxxxxxxxxxxxxx     | -1.000000                   | 음의 최소값으로 포화 |


## ✂️ Truncation (절삭)

- **정의**: 고정소수점 수에서 표현 비트 수를 줄일 때, 하위 비트(덜 중요한 비트)를 단순히 잘라내는 방법.
- **목적**: 소수점 자리수를 줄여서 비트 수를 맞추기 위해 사용.
- **특징**: 버림 방식이기 때문에 약간의 값 손실(precision loss)이 발생할 수 있음.

---