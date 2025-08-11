---
title: "System Verolog Day-4"
date: "2025-08-11"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# RISC vs CISC 차이

## 1. 설계 철학

| 구분 | RISC (Reduced Instruction Set Computer) | CISC (Complex Instruction Set Computer) |
|------|----------------------------------------|-----------------------------------------|
| 핵심 아이디어 | 간단하고 빠른 명령어를 적게 제공 → 실행 속도 향상, 파이프라이닝 최적화 | 다양한 복잡한 명령어 제공 → 한 명령으로 많은 작업 수행 |
| 목표 | 하드웨어 단순화, 클럭당 실행 명령 수 향상 | 프로그램 코드 길이 단축, 소프트웨어 개발 단순화(과거 기준) |

---

## 2. 명령어 구조

| 구분 | RISC | CISC |
|------|------|------|
| 명령어 개수 | 적음 (수십 개 수준) | 많음 (수백 개 이상) |
| 명령어 길이 | 고정 길이 (예: 32비트) → 디코딩 단순 | 가변 길이 (1~15바이트 등) → 디코딩 복잡 |
| 실행 시간 | 대부분 1클럭 (load/store 제외) | 명령어마다 실행 클럭 수 다름 |
| 주소 지정 방식 | 소수 (Load/Store 중심) | 다양 (메모리 직접 연산 가능) |
| 메모리 접근 | Load/Store 방식 (메모리 접근은 별도 명령) | 명령어가 직접 메모리 접근 가능 |

---

## 3. 하드웨어와 소프트웨어 역할

- **RISC**
  - 하드웨어 단순
  - 명령어 단순 → 복잡한 기능은 소프트웨어(컴파일러)가 여러 명령으로 조합
  - 파이프라인 효율 높음
- **CISC**
  - 하드웨어 복잡
  - 명령어 자체가 여러 연산 포함
  - 어셈블리 코드가 짧아짐 (메모리 절약 장점)

---

## 4. 대표 예시

| RISC | CISC |
|------|------|
| ARM, MIPS, RISC-V, SPARC, PowerPC | x86, x86-64, VAX, Motorola 68000 |

---

## 5. 요약 비교표

| 특징 | RISC | CISC |
|------|------|------|
| 명령어 수 | 적음 | 많음 |
| 명령어 길이 | 고정 | 가변 |
| 실행 속도 | 빠름 (단순 연산 다수) | 느림 (복잡 연산 일부) |
| 파이프라인 효율 | 높음 | 낮음 |
| 하드웨어 복잡도 | 낮음 | 높음 |
| 소프트웨어 복잡도 | 높음 | 낮음(과거 기준) |

---

## 💡 정리

- **RISC**: "간단한 명령어 + 빠른 실행 + 파이프라인 최적화" → 현대 모바일·임베디드 CPU(ARM, RISC-V)에서 주로 사용
- **CISC**: "복잡한 명령어 + 코드 길이 절약" → 데스크톱·서버 CPU(x86)에서 강세
- 최신 x86 CPU는 내부적으로 RISC와 유사한 **마이크로-오퍼레이션(uOp) 변환** 구조 사용 → 두 구조의 경계가 흐려짐


# 폰 노이만 구조 vs 하버드 구조

## 1. 개념

| 구조 | 개념 |
|------|------|
| **폰 노이만 구조** | 프로그램 코드와 데이터를 **같은 메모리**에 저장, 하나의 버스를 통해 CPU와 메모리를 연결 |
| **하버드 구조** | 프로그램 코드와 데이터를 **서로 다른 메모리**에 저장, 각각 독립된 버스를 사용 |

---

## 2. 구조도

### 폰 노이만 & 하버드 구조

![alt text](../../../assets/img/CPU/day_4/image.png)

**폰 노이만 구조**
![alt text](<../../../assets/img/CPU/day_4/스크린샷 2025-08-11 125833.png>)

**하버드 구조**
![alt text](<../../../assets/img/CPU/day_4/스크린샷 2025-08-11 125920.png>)

## 3. 특징 비교

| 구분 | 폰 노이만 구조 | 하버드 구조 |
|------|---------------|-------------|
| 메모리 구성 | 코드와 데이터가 같은 메모리 | 코드와 데이터가 분리된 메모리 |
| 버스 구조 | 단일 버스 | 명령어 버스와 데이터 버스 분리 |
| 동시 접근 | 불가능 (한 번에 명령어 또는 데이터) | 가능 (명령어와 데이터 동시에 접근) |
| 하드웨어 복잡도 | 단순 | 복잡 |
| 속도 | 상대적으로 느림 (버스 병목, Von Neumann bottleneck) | 빠름 (병목 완화) |
| 예시 | 대부분의 범용 CPU(x86, ARM의 대부분) | DSP, 일부 마이크로컨트롤러(AVR, PIC 등) |

---

## 4. 정리

- **폰 노이만 구조**: 단순하고 구현이 쉬움 → 범용 컴퓨터, 서버, PC에서 주로 사용  
- **하버드 구조**: 명령어와 데이터를 동시에 가져올 수 있어 성능 우수 → 실시간 처리, 임베디드, DSP에서 활용  
- 현대 CPU는 **수정된 하버드 구조(Modified Harvard)**를 사용하여 캐시 단계에서 분리 후 통합



# 수업

## C언어로 0~9를 count 하는 processor
![alt text](<../../../assets/img/CPU/day_4/스크린샷 2025-08-11 145121.png>)

```verilog
`timescale 1ns / 1ps

module DedicatedProcessor_Counter (
    input  logic       clk,
    input  logic       reset,
    output logic [7:0] OutBuffer
);

    logic ASrcMuxSel;
    logic AEn;
    logic ALt10;
    logic OutBufEn;

    logic [$clog2(10_000_000) -1:0] div_counter;
    logic clk_10hz;

    //////////////////////// clk_div ////////////////////////
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            div_counter <= 0;
        end else begin
            if (div_counter == 10_000_000 - 1) begin
                div_counter <= 0;
                clk_10hz <= 1'b1;
            end else begin
                div_counter <= div_counter + 1;
                clk_10hz <= 1'b0;
            end
        end
    end

    ControlUnit U_ControlUnit (
        .clk(clk_10hz),
        .*
    );
    DataPath U_DataPath (
        .clk(clk_10hz),
        .*
    );
endmodule

/////////////////////////////////////////////////////////////////////////

module DataPath (
    input  logic       clk,
    input  logic       reset,
    input  logic       ASrcMuxSel,
    input  logic       AEn,
    output logic       ALt10,
    input  logic       OutBufEn,
    output logic [7:0] OutBuffer
);
    logic [7:0] adderResult, ASrcMuxOut, ARegOut;

    mux_2x1 U_ASrcNux (
        .sel(ASrcMuxSel),
        .x0 (8'b0),
        .x1 (adderResult),
        .y  (ASrcMuxOut)
    );

    register U_A_Reg (
        .clk  (clk),
        .reset(reset),
        .en   (AEn),
        .d    (ASrcMuxOut),
        .q    (ARegOut)
    );

    comparator U_ALt10 (
        .a (ARegOut),
        .b (8'd10),
        .lt(ALt10)
    );

    adder U_Adder (
        .a  (ARegOut),
        .b  (8'd1),
        .sum(adderResult)
    );

    OutBuf U_OutBuf (
        .en(OutBufEn),
        .x (ARegOut),
        .y (OutBuffer)
    );

    // register U_OutReg(
    //     .clk  (clk),
    //     .reset(reset),
    //     .en   (OutBufEn),
    //     .d    (ARegOut),
    //     .q    (OutBuffer)
    // );

endmodule

/////////////////////////////////////////////////////////////////////////

module register (
    input  logic       clk,
    input  logic       reset,
    input  logic       en,
    input  logic [7:0] d,
    output logic [7:0] q
);
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            q <= 0;
        end else begin
            if (en) begin
                q <= d;
            end
        end
    end
endmodule


module mux_2x1 (
    input  logic       sel,
    input  logic [7:0] x0,
    input  logic [7:0] x1,
    output logic [7:0] y
);
    always_comb begin
        y = 8'b0;
        case (sel)
            1'b0: y = x0;
            1'b1: y = x1;
        endcase
    end
endmodule


module adder (
    input  logic [7:0] a,
    input  logic [7:0] b,
    output logic [7:0] sum
);
    assign sum = a + b;
endmodule


module comparator (
    input  logic [7:0] a,
    input  logic [7:0] b,
    output logic       lt
);
    assign lt = a < b;
endmodule


module OutBuf (
    input  logic       en,
    input  logic [7:0] x,
    output logic [7:0] y
);
    assign y = en ? x : 8'bx;
endmodule

/////////////////////////////////////////////////////////////////////////

module ControlUnit (
    input  logic clk,
    input  logic reset,
    output logic ASrcMuxSel,
    output logic AEn,
    input  logic ALt10,
    output logic OutBufEn
);
    typedef enum {
        S0,
        S1,
        S2,
        S3,
        S4
    } state_e;

    state_e state, next_state;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            state <= S0;
        end else begin
            state <= next_state;
        end
    end

    always_comb begin
        ASrcMuxSel = 0;
        AEn = 0;
        OutBufEn = 0;
        next_state = state;
        case (state)
            S0: begin
                ASrcMuxSel = 0;
                AEn = 1;
                OutBufEn = 0;
                next_state = S1;
            end
            S1: begin
                ASrcMuxSel = 1;
                AEn = 0;
                OutBufEn = 0;
                if (ALt10) begin
                    next_state = S2;
                end else begin
                    next_state = S4;
                end
            end
            S2: begin
                ASrcMuxSel = 1;
                AEn = 0;
                OutBufEn = 1;
                next_state = S3;
            end
            S3: begin
                ASrcMuxSel = 1;
                AEn = 1;
                OutBufEn = 0;
                next_state = S1;
            end
            S4: begin
                ASrcMuxSel = 1;
                AEn = 0;
                OutBufEn = 0;
                next_state = S4;
            end
        endcase
    end
endmodule
```


# Home Work
**0~55까지 0~10을 전부 더하라**
![alt text](<../../../assets/img/CPU/day_4/스크린샷 2025-08-11 184435.png>)

### DedicatedProcessor_Sum.sv
```verilog
`timescale 1ns / 1ps

module DedicatedProcessor_Sum (
    input  logic       clk,
    input  logic       reset,
    output logic [7:0] OutBuffer
);

    logic ASrcMuxSel;
    logic AEn;
    logic ALt10;
    logic OutBufEn;

    logic [$clog2(10_000_000) -1:0] div_counter;
    logic clk_10hz;

    //////////////////////// clk_div ////////////////////////
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            div_counter <= 0;
        end else begin
            if (div_counter == 10_000_000 - 1) begin
                div_counter <= 0;
                clk_10hz <= 1'b1;
            end else begin
                div_counter <= div_counter + 1;
                clk_10hz <= 1'b0;
            end
        end
    end

    ControlUnit U_ControlUnit (
        .clk(clk_10hz),
        .*
    );
    DataPath U_DataPath (
        .clk(clk_10hz),
        .*
    );
endmodule

/////////////////////////////////////////////////////////////////////////

module DataPath (
    input  logic       clk,
    input  logic       reset,
    input  logic       ASrcMuxSel,
    input  logic       AEn,
    output logic       ALt10,
    input  logic       OutBufEn,
    output logic [7:0] OutBuffer
);
    logic [7:0] adderResult_A, A_SrcMuxOut, A_RegOut;
    logic [7:0] S_SrcMuxOut, S_RegOut;
    logic [7:0] adderResult_S;

    mux_2x1 U_ASrcMux (
        .sel(ASrcMuxSel),
        .x0 (8'b0),
        .x1 (adderResult_A),
        .y  (A_SrcMuxOut)
    );

    register U_A_Reg (
        .clk  (clk),
        .reset(reset),
        .en   (AEn),
        .d    (A_SrcMuxOut),
        .q    (A_RegOut)
    );

    comparator U_ALte10 (
        .a (A_RegOut),
        .b (8'd11),
        .lte(ALt10)
    );

    adder U_Adder_A (
        .a  (A_RegOut),
        .b  (8'd1),
        .sum(adderResult_A)
    );

    mux_2x1 U_SSrcMux (
        .sel(ASrcMuxSel),
        .x0 (8'b0),
        .x1 (adderResult_S),
        .y  (S_SrcMuxOut)
    );

    register U_S_Reg (
        .clk  (clk),
        .reset(reset),
        .en   (AEn),
        .d    (S_SrcMuxOut),
        .q    (S_RegOut)
    );

    adder U_Adder_Sum (
        .a  (S_RegOut),
        .b  (adderResult_A),
        .sum(adderResult_S)
    );

    // OutBuf U_OutBuf (
    //     .en(OutBufEn),
    //     .x (S_RegOut),
    //     .y (OutBuffer)
    // );

    register U_OutReg(
        .clk  (clk),
        .reset(reset),
        .en   (OutBufEn),
        .d    (S_RegOut),
        .q    (OutBuffer)
    );

endmodule

/////////////////////////////////////////////////////////////////////////

module register (
    input  logic       clk,
    input  logic       reset,
    input  logic       en,
    input  logic [7:0] d,
    output logic [7:0] q
);
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            q <= 0;
        end else begin
            if (en) begin
                q <= d;
            end
        end
    end
endmodule


module mux_2x1 (
    input  logic       sel,
    input  logic [7:0] x0,
    input  logic [7:0] x1,
    output logic [7:0] y
);
    always_comb begin
        y = 8'b0;
        case (sel)
            1'b0: y = x0;
            1'b1: y = x1;
        endcase
    end
endmodule


module adder (
    input  logic [7:0] a,
    input  logic [7:0] b,
    output logic [7:0] sum
);
    assign sum = a + b;
endmodule


module comparator (
    input  logic [7:0] a,
    input  logic [7:0] b,
    output logic       lte
);
    assign lte = a < b;
endmodule


module OutBuf (
    input  logic       en,
    input  logic [7:0] x,
    output logic [7:0] y
);
    assign y = en ? x : 8'bx;
endmodule

/////////////////////////////////////////////////////////////////////////

module ControlUnit (
    input  logic clk,
    input  logic reset,
    output logic ASrcMuxSel,
    output logic AEn,
    input  logic ALt10,
    output logic OutBufEn
);
    typedef enum {
        S0,
        S1,
        S2,
        S3,
        S4
    } state_e;

    state_e state, next_state;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            state <= S0;
        end else begin
            state <= next_state;
        end
    end

    always_comb begin
        ASrcMuxSel = 0;
        AEn = 0;
        OutBufEn = 0;
        next_state = state;
        case (state)
            S0: begin
                ASrcMuxSel = 0;
                AEn = 1;
                OutBufEn = 0;
                next_state = S1;
            end
            S1: begin
                ASrcMuxSel = 1;
                AEn = 0;
                OutBufEn = 0;
                if (ALt10) begin
                    next_state = S2;
                end else begin
                    next_state = S4;
                end
            end
            S2: begin
                ASrcMuxSel = 1;
                AEn = 0;
                OutBufEn = 1;
                next_state = S3;
            end
            S3: begin
                ASrcMuxSel = 1;
                AEn = 1;
                OutBufEn = 0;
                next_state = S1;
            end
            S4: begin
                ASrcMuxSel = 1;
                AEn = 0;
                OutBufEn = 0;
                next_state = S4;
            end
        endcase
    end
endmodule
```

### 결과 
![alt text](<../../../assets/img/CPU/day_4/스크린샷 2025-08-11 182104.png>)
<video controls src="../../../assets/img/CPU/day_4/영상.mp4" title="Title"></video>