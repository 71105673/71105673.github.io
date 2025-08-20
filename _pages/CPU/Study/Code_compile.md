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
C 언어에서 Return은 JALR을 사용

## 주의점
해당 방식과 같이 sp 의 주소 또한 초기화 해 줘야한다.
```bash
		li		sp, 0x40   
main:
        addi    sp,sp,-32
        sw      ra,28(sp)
        sw      s0,24(sp)
        addi    s0,sp,32
        li      a5,10
        sw      a5,-20(s0)
        li      a5,20
        sw      a5,-24(s0)
        lw      a1,-24(s0)
        lw      a0,-20(s0)
        call    adder
        sw      a0,-28(s0)
        li      a5,0
        mv      a0,a5
        lw      ra,28(sp)
        lw      s0,24(sp)
        addi    sp,sp,32
        jr      ra
adder:
        addi    sp,sp,-32
        sw      ra,28(sp)
        sw      s0,24(sp)
        addi    s0,sp,32
        sw      a0,-20(s0)
        sw      a1,-24(s0)
        lw      a4,-20(s0)
        lw      a5,-24(s0)
        add     a5,a4,a5
        mv      a0,a5
        lw      ra,28(sp)
        lw      s0,24(sp)
        addi    sp,sp,32
        jr      ra
```

## 분석

![alt text](../../../assets/img/CPU/code_com/c_ass.png)
![alt text](../../../assets/img/CPU/code_com/memory구조.png)
![alt text](<../../../assets/img/CPU/code_com/스크린샷 2025-08-20 142858.png>)



# Home Work

## rom 쓰는 법
Create memory file을 같은 경로에 넣는다.
후에 code에서 다음과 같이 Hexa 형식으로 읽어 rom에 저장할 수 있다.
![alt text](<../../../assets/img/CPU/code_com/스크린샷 2025-08-20 151441.png>)

### C
```C
void sort(int *pData, int size);
void swap(int *pA, int *pB);

int main(){
    int arData[6] = {5, 4, 3, 2, 1};

    sort(arData, 5);

    return 0;
}

void sort(int *pData, int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size - i - 1; j++){
            if (pData[j] > pData[j + 1])
                swap(&pData[j], &pData[j+1]);  
        }
    }
}

void swap(int *pA, int *pB) {
    int temp;
    temp = *pA;
    *pA = *pB;
    *pB = temp;
}
```

### Assembly
```bash
main:
        addi    sp,sp,-48
        sw      ra,44(sp)
        sw      s0,40(sp)
        addi    s0,sp,48
        sw      zero,-40(s0)
        sw      zero,-36(s0)
        sw      zero,-32(s0)
        sw      zero,-28(s0)
        sw      zero,-24(s0)
        sw      zero,-20(s0)
        li      a5,5
        sw      a5,-40(s0)
        li      a5,4
        sw      a5,-36(s0)
        li      a5,3
        sw      a5,-32(s0)
        li      a5,2
        sw      a5,-28(s0)
        li      a5,1
        sw      a5,-24(s0)
        addi    a5,s0,-40
        li      a1,5
        mv      a0,a5
        call    sort
        li      a5,0
        mv      a0,a5
        lw      ra,44(sp)
        lw      s0,40(sp)
        addi    sp,sp,48
        jr      ra
sort:
        addi    sp,sp,-48
        sw      ra,44(sp)
        sw      s0,40(sp)
        addi    s0,sp,48
        sw      a0,-36(s0)
        sw      a1,-40(s0)
        sw      zero,-20(s0)
        j       .L4
.L8:
        sw      zero,-24(s0)
        j       .L5
.L7:
        lw      a5,-24(s0)
        slli    a5,a5,2
        lw      a4,-36(s0)
        add     a5,a4,a5
        lw      a4,0(a5)
        lw      a5,-24(s0)
        addi    a5,a5,1
        slli    a5,a5,2
        lw      a3,-36(s0)
        add     a5,a3,a5
        lw      a5,0(a5)
        ble     a4,a5,.L6
        lw      a5,-24(s0)
        slli    a5,a5,2
        lw      a4,-36(s0)
        add     a3,a4,a5
        lw      a5,-24(s0)
        addi    a5,a5,1
        slli    a5,a5,2
        lw      a4,-36(s0)
        add     a5,a4,a5
        mv      a1,a5
        mv      a0,a3
        call    swap
.L6:
        lw      a5,-24(s0)
        addi    a5,a5,1
        sw      a5,-24(s0)
.L5:
        lw      a4,-40(s0)
        lw      a5,-20(s0)
        sub     a5,a4,a5
        addi    a5,a5,-1
        lw      a4,-24(s0)
        blt     a4,a5,.L7
        lw      a5,-20(s0)
        addi    a5,a5,1
        sw      a5,-20(s0)
.L4:
        lw      a4,-20(s0)
        lw      a5,-40(s0)
        blt     a4,a5,.L8
        nop
        nop
        lw      ra,44(sp)
        lw      s0,40(sp)
        addi    sp,sp,48
        jr      ra
swap:
        addi    sp,sp,-48
        sw      ra,44(sp)
        sw      s0,40(sp)
        addi    s0,sp,48
        sw      a0,-36(s0)
        sw      a1,-40(s0)
        lw      a5,-36(s0)
        lw      a5,0(a5)
        sw      a5,-20(s0)
        lw      a5,-40(s0)
        lw      a4,0(a5)
        lw      a5,-36(s0)
        sw      a4,0(a5)
        lw      a5,-40(s0)
        lw      a4,-20(s0)
        sw      a4,0(a5)
        nop
        lw      ra,44(sp)
        lw      s0,40(sp)
        addi    sp,sp,48
        jr      ra
```