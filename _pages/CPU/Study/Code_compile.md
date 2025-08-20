---
title: "code_compile"
date: "2025-08-20"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# Link

## C -> Assembly
https://godbolt.org/
## Assembly
https://riscvasm.lucasteske.dev/
## RISC-V 분석기(machine code)
https://luplab.gitlab.io/rvcodecjs/

첫 사이트에서 C언어 코드를 입력하면 어셈블리어로 자동으로 변경된다.
![text](<../../../assets/img/CPU/code_com/스크린샷 2025-08-20 143338.png>) 

2번째 사이트에서 왼쪽에 어셈블리어를 입력하고 Build를 하게 되면.
아래와 같이 결과가 나온다.
![text](<../../../assets/img/CPU/code_com/스크린샷 2025-08-20 143359.png>) 

3번째 부분의 코드를 보게 되면 머신코드가 어떤 부분 어셈블리어에서 변경되었는지 알 수 있다.

C -> ![alt text](<../../../assets/img/CPU/code_com/스크린샷 2025-08-20 143737.png>)

Assembly -> ![alt text](<../../../assets/img/CPU/code_com/스크린샷 2025-08-20 143733.png>)

machine code -> ![alt text](<../../../assets/img/CPU/code_com/스크린샷 2025-08-20 143710.png>)


![text](<../../../assets/img/CPU/code_com/스크린샷 2025-08-20 143407.png>)



# 분석

![alt text](../../../assets/img/CPU/code_com/c_ass.png)
![alt text](../../../assets/img/CPU/code_com/memory구조.png)
![alt text](<../../../assets/img/CPU/code_com/스크린샷 2025-08-20 142858.png>)