---
title: "Project_1 Sysyem Verilog" 
date: "2025-07-16"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

> 대부분 Low Pass Filter 사용
![alt text](<../../../assets/img/SystemVerilog/project_1/스크린샷 2025-07-15 161918.png>)

> 컨볼루션 개념
![alt text](<../../../assets/img/SystemVerilog/project_1/스크린샷 2025-07-15 162201.png>)

> 여기서 z-1은 shift reg라 생각하면 편함
![alt text](<../../../assets/img/SystemVerilog/project_1/스크린샷 2025-07-15 162632.png>)

> 고정소숫점 방식 -> 비트수가 작으면 노이즈가 증가, 그러나 area는 작음
> 
> 이것을 fixed point simulation
![alt text](<../../../assets/img/SystemVerilog/project_1/스크린샷 2025-07-15 163210.png>)

> RRC Filter -> 주파수 도메인 확인
>
> 여기서 fixed = 1.8인 이유가 정수부가 작아서
>
> 즉 정수부 1, 소수부 8 = 9bit
![alt text](<../../../assets/img/SystemVerilog/project_1/스크린샷 2025-07-15 163719.png>)