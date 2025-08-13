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

![alt text](<../../../assets/img/CPU/day_5/스크린샷 2025-08-12 104703.png>)

## top_sum.sv
```verilog
module ControlUnit (
    input  logic       clk,
    input  logic       reset,
    input  logic       ILte10,
    input  logic [7:0] OutPort,
    output logic       SumSrcMuxSel,
    output logic       ISrcMuxSel,
    output logic       SumEn,
    output logic       IEn,
    output logic       AdderSrcMuxSel,
    output logic       OutPortEn
);
    typedef enum {
        S0,
        S1,
        S2,
        S3,
        S4,
        S5
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
        next_state     = state;
        SumSrcMuxSel   = 0;
        ISrcMuxSel     = 0;
        SumEn          = 0;
        IEn            = 0;
        AdderSrcMuxSel = 0;
        OutPortEn      = 0;
        case (state)
            S0: begin
                SumSrcMuxSel   = 0;
                ISrcMuxSel     = 0;
                SumEn          = 1;
                IEn            = 1;
                AdderSrcMuxSel = 0;
                OutPortEn      = 0;
                next_state     = S1;
            end
            S1: begin
                SumSrcMuxSel   = 0;
                ISrcMuxSel     = 0;
                SumEn          = 0;
                IEn            = 0;
                AdderSrcMuxSel = 0;
                OutPortEn      = 0;
                if (ILte10) begin
                    next_state = S2;
                end else begin
                    next_state = S5;
                end
            end
            S2: begin
                SumSrcMuxSel   = 1;
                ISrcMuxSel     = 1;
                SumEn          = 1;
                IEn            = 0;
                AdderSrcMuxSel = 0;
                OutPortEn      = 0;
                next_state     = S3;
            end
            S3: begin
                SumSrcMuxSel   = 1;
                ISrcMuxSel     = 1;
                SumEn          = 0;
                IEn            = 1;
                AdderSrcMuxSel = 1;
                OutPortEn      = 0;
                next_state     = S4;
            end
            S4: begin
                SumSrcMuxSel   = 1;
                ISrcMuxSel     = 1;
                SumEn          = 0;
                IEn            = 0;
                AdderSrcMuxSel = 0;
                OutPortEn      = 1;
                next_state     = S1;
            end
            S5: begin
                SumSrcMuxSel   = 1;
                ISrcMuxSel     = 1;
                SumEn          = 0;
                IEn            = 0;
                AdderSrcMuxSel = 0;
                OutPortEn      = 0;
                next_state     = S5;
            end
        endcase
    end
endmodule
```

## DedicatedProcessor_Sum.sv
```verilog
`timescale 1ns / 1ps

module DedicatedProcessor_Sum (
    input  logic       clk,
    input  logic       reset,
    output logic [7:0] OutPort
);
    logic SumSrcMuxSel;
    logic ISrcMuxSel;
    logic SumEn;
    logic IEn;
    logic AdderSrcMuxSel;
    logic OutPortEn;
    logic ILte10;

    DataPath U_DP (.*);

    ControlUnit U_CU (.*);

endmodule
```

## ControlUnit.sv
```verilog
module ControlUnit (
    input  logic       clk,
    input  logic       reset,
    input  logic       ILte10,
    input  logic [7:0] OutPort,
    output logic       SumSrcMuxSel,
    output logic       ISrcMuxSel,
    output logic       SumEn,
    output logic       IEn,
    output logic       AdderSrcMuxSel,
    output logic       OutPortEn
);
    typedef enum {
        S0,
        S1,
        S2,
        S3,
        S4,
        S5
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
        next_state     = state;
        SumSrcMuxSel   = 0;
        ISrcMuxSel     = 0;
        SumEn          = 0;
        IEn            = 0;
        AdderSrcMuxSel = 0;
        OutPortEn      = 0;
        case (state)
            S0: begin
                SumSrcMuxSel   = 0;
                ISrcMuxSel     = 0;
                SumEn          = 1;
                IEn            = 1;
                AdderSrcMuxSel = 0;
                OutPortEn      = 0;
                next_state     = S1;
            end
            S1: begin
                SumSrcMuxSel   = 0;
                ISrcMuxSel     = 0;
                SumEn          = 0;
                IEn            = 0;
                AdderSrcMuxSel = 0;
                OutPortEn      = 0;
                if (ILte10) begin
                    next_state = S2;
                end else begin
                    next_state = S5;
                end
            end
            S2: begin
                SumSrcMuxSel   = 1;
                ISrcMuxSel     = 1;
                SumEn          = 1;
                IEn            = 0;
                AdderSrcMuxSel = 0;
                OutPortEn      = 0;
                next_state     = S3;
            end
            S3: begin
                SumSrcMuxSel   = 1;
                ISrcMuxSel     = 1;
                SumEn          = 0;
                IEn            = 1;
                AdderSrcMuxSel = 1;
                OutPortEn      = 0;
                next_state     = S4;
            end
            S4: begin
                SumSrcMuxSel   = 1;
                ISrcMuxSel     = 1;
                SumEn          = 0;
                IEn            = 0;
                AdderSrcMuxSel = 0;
                OutPortEn      = 1;
                next_state     = S1;
            end
            S5: begin
                SumSrcMuxSel   = 1;
                ISrcMuxSel     = 1;
                SumEn          = 0;
                IEn            = 0;
                AdderSrcMuxSel = 0;
                OutPortEn      = 0;
                next_state     = S5;
            end
        endcase
    end
endmodule
```

## DataPath.sv
```verilog
`timescale 1ns / 1ps

module DataPath (
    input  logic       clk,
    input  logic       reset,
    input  logic       SumSrcMuxSel,
    input  logic       ISrcMuxSel,
    input  logic       SumEn,
    input  logic       IEn,
    input  logic       AdderSrcMuxSel,
    input  logic       OutPortEn,
    output logic       ILte10,
    output logic [7:0] OutPort
);

    logic [7:0] SumSrcMuxOut, SumRegOut;
    logic [7:0] ISrcMuxOut, IRegOut;
    logic [7:0] AdderResult, AdderSrcMuxOut;

    mux_2x1 U_SumSrcMux (
        .sel(SumSrcMuxSel),
        .x0 (0),
        .x1 (AdderResult),
        .y  (SumSrcMuxOut)
    );

    mux_2x1 U_ISrcMux (
        .sel(ISrcMuxSel),
        .x0 (0),
        .x1 (AdderResult),
        .y  (ISrcMuxOut)
    );

    register Sum_Reg (
        .clk  (clk),
        .reset(reset),
        .en   (SumEn),
        .d    (SumSrcMuxOut),
        .q    (SumRegOut)
    );

    register I_Reg (
        .clk  (clk),
        .reset(reset),
        .en   (IEn),
        .d    (ISrcMuxOut),
        .q    (IRegOut)
    );

    comparator I_Lte10 (
        .a  (IRegOut),
        .b  (10),
        .lte(ILte10)
    );

    mux_2x1 U_AdderSrcMux (
        .sel(AdderSrcMuxSel),
        .x0 (SumRegOut),
        .x1 (1),
        .y  (AdderSrcMuxOut)
    );

    adder U_Adder (
        .a  (AdderSrcMuxOut),
        .b  (IRegOut),
        .sum(AdderResult)
    );

    register U_OutPort (
        .clk  (clk),
        .reset(reset),
        .en   (OutPortEn),
        .d    (SumRegOut),
        .q    (OutPort)
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
    assign lte = a <= b;
endmodule


// module OutBuf (
//     input  logic       en,
//     input  logic [7:0] x,
//     output logic [7:0] y
// );
//     assign y = en ? x : 8'bx;
// endmodule
```

![alt text](<../../../assets/img/CPU/day_5/스크린샷 2025-08-12 132915.png>)


# Home Work 1
**Register File** 을 만들어, 동작 기능

![alt text](<../../../assets/img/CPU/day_5/화면 캡처 2025-08-12 155354.png>)
![alt text](<../../../assets/img/CPU/day_5/화면 캡처 2025-08-12 155320.png>)

# Home Work 2
**ALU** 기능을 만들어 **ALUop 2bit** 를 통해 +, -, &, | 연산을 추가하기
![alt text](<../../../assets/img/CPU/day_5/스크린샷 2025-08-12 162026.png>)

## ASM
![alt text](<../../../assets/img/CPU/day_5/스크린샷 2025-08-13 085755.png>)

## 예측 결과 
```
# 반복별 요약(각 반복의 마지막 상태)

* 1회: R1=3,  R2=2,  R3=3,  R4=5   → `R4>R2` Yes
* 2회: R1=7,  R2=6,  R3=9,  R4=15  → Yes
* 3회: R1=15, R2=14, R3=21, R4=35  → Yes
* 4회: R1=31, R2=30, R3=45, R4=75  → Yes
* 5회: R1=63, R2=62, R3=93, R4=155 → Yes
* 6회: R1=127,R2=126,R3=189,R4=59  → **No ⇒ halt**

# 전체 계산 로그(각 단계 값)

**반복 1**

* start: (R1, R2, R3, R4) = (1, 0, 0, 0)
* R2 = R1+R1 → (1, 2, 0, 0)
* R3 = R2+R1 → (1, 2, 3, 0)
* R4 = R3−R1 → (1, 2, 3, 2)
* R1 = R1 | R2 → (3, 2, 3, 2)
* (첫 비교 `R4<R2`는 통과만)
* R4 = R4 & R3 → (3, 2, 3, 2)
* R4 = R2+R3 → (3, 2, 3, 5)
* 체크 `R4>R2` = 5>2 → **Yes**

**반복 2**

* start: (3, 2, 3, 5)
* R2=R1+R1 → (3, 6, 3, 5)
* R3=R2+R1 → (3, 6, 9, 5)
* R4=R3−R1 → (3, 6, 9, 6)
* R1=R1|R2 → (7, 6, 9, 6)
* R4=R4\&R3 → (7, 6, 9, 0x06&0x09=0x00? 아님 6&9=0) → (7, 6, 9, 0)
* R4=R2+R3 → (7, 6, 9, 15)
* 체크 15>6 → **Yes**

**반복 3**

* start: (7, 6, 9, 15)
* R2=14 → (7,14,9,15)
* R3=21 → (7,14,21,15)
* R4=14 → (7,14,21,14)
* R1=15 → (15,14,21,14)
* R4=14&21=4 → (15,14,21,4)
* R4=14+21=35 → (15,14,21,35)
* 체크 35>14 → **Yes**

**반복 4**

* start: (15,14,21,35)
* R2=30 → (15,30,21,35)
* R3=45 → (15,30,45,35)
* R4=30 → (15,30,45,30)
* R1=31 → (31,30,45,30)
* R4=30&45=12 → (31,30,45,12)
* R4=30+45=75 → (31,30,45,75)
* 체크 75>30 → **Yes**

**반복 5**

* start: (31,30,45,75)
* R2=62 → (31,62,45,75)
* R3=93 → (31,62,93,75)
* R4=62 → (31,62,93,62)
* R1=63 → (63,62,93,62)
* R4=62&93=28 → (63,62,93,28)
* R4=62+93=155 → (63,62,93,155)
* 체크 155>62 → **Yes**

**반복 6**

* start: (63,62,93,155)
* R2=126 → (63,126,93,155)
* R3=219 → **8비트 래핑 →** 219 (0xDB) → (63,126,219,155)
* R4=219−63=156 → (63,126,219,156)
* R1=127 → (127,126,219,156)
* R4=156&219=156&0xDB=0x98(=152) → (127,126,219,152)
* R4=126+219=345 → 8비트 래핑 → 345−256=89 → 실제 126+189

  * 정확히는 R3가 219가 아니라 189 (126+63=189) 따라서 R4=126+189=315 → 315−256 = 59
    → (127,126,189,59)
* 59>126 → No ⇒ halt

> Halt 되는 이유:
> 매 반복에서 `R4 = (R2 + R3) mod 256`이라, 값이 커지다가 6회차에 오버플로우가 나서 R4=59로 작아져 R4>R2가 거짓이 돼서 멈춤
```

## DedicatedProcessor.sv
```verilog
`timescale 1ns / 1ps

module DedicatedProcessor (
    input logic clk,
    input logic reset,
    output logic [7:0] OutPort
);
    logic       RFSrcMuxSel;
    logic [2:0] readAddr1;
    logic [2:0] readAddr2;
    logic [2:0] writeAddr;
    logic       writeEn;
    logic [1:0] aluOP;
    logic       R1Le10;
    logic       OutPortEn;

    ControlUnit U_CU (.*);
    DataPath U_DP (.*);
endmodule
```

## ControlUnit
```verilog
`timescale 1ns / 1ps

module ControlUnit (
    input  logic       clk,
    input  logic       reset,
    input  logic       R1Le10,
    output logic       RFSrcMuxSel,
    output logic [2:0] readAddr1,
    output logic [2:0] readAddr2,
    output logic [2:0] writeAddr,
    output logic       writeEn,
    output logic [1:0] aluOP,
    output logic       OutPortEn
);

    typedef enum {
        S0,
        S1,
        S2,
        S3,
        S4,
        S5,
        S6,
        S7,
        S8,
        S9,
        S10,
        S11,
        S12,
        HALT
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
        next_state  = state;
        RFSrcMuxSel = 0;
        readAddr1   = 0;
        readAddr2   = 0;
        writeAddr   = 0;
        writeEn     = 0;
        aluOP       = 0;
        OutPortEn   = 0;
        case (state)
            S0: begin  // R1 = 1
                RFSrcMuxSel = 1;
                readAddr1   = 0;
                readAddr2   = 0;
                writeAddr   = 1;
                writeEn     = 1;
                aluOP       = 0;
                OutPortEn   = 0;
                next_state  = S1;
            end
            S1: begin  // R2 = 0
                RFSrcMuxSel = 0;
                readAddr1   = 0;
                readAddr2   = 0;
                writeAddr   = 2;
                writeEn     = 1;
                aluOP       = 0;
                OutPortEn   = 0;
                next_state  = S2;
            end
            S2: begin  // R3 = 0
                RFSrcMuxSel = 0;
                readAddr1   = 0;
                readAddr2   = 0;
                writeAddr   = 3;
                writeEn     = 1;
                aluOP       = 0;
                OutPortEn   = 0;
                next_state  = S3;
            end
            S3: begin  // R4 = 0
                RFSrcMuxSel = 0;
                readAddr1   = 0;
                readAddr2   = 0;
                writeAddr   = 4;
                writeEn     = 1;
                aluOP       = 0;
                OutPortEn   = 0;
                next_state  = S4;
            end
            S4: begin  // R2 = R1 + R1
                RFSrcMuxSel = 0;
                readAddr1   = 1;
                readAddr2   = 1;
                writeAddr   = 2;
                writeEn     = 1;
                aluOP       = 0;
                OutPortEn   = 0;
                next_state  = S5;
            end
            S5: begin  // R3 = R2 + R1
                RFSrcMuxSel = 0;
                readAddr1   = 2;
                readAddr2   = 1;
                writeAddr   = 3;
                writeEn     = 1;
                aluOP       = 0;
                OutPortEn   = 0;
                next_state  = S6;
            end
            S6: begin  // R4 = R3 - R1
                RFSrcMuxSel = 0;
                readAddr1   = 3;
                readAddr2   = 1;
                writeAddr   = 4;
                writeEn     = 1;
                aluOP       = 1;
                OutPortEn   = 0;
                next_state  = S7;
            end
            S7: begin  // R1 = R1 | R2
                RFSrcMuxSel = 0;
                readAddr1   = 1;
                readAddr2   = 2;
                writeAddr   = 1;
                writeEn     = 1;
                aluOP       = 3;
                OutPortEn   = 0;
                next_state  = S8;
            end
            S8: begin  // R4 < R2
                RFSrcMuxSel = 0;
                readAddr1   = 2;
                readAddr2   = 4;
                writeAddr   = 1;
                writeEn     = 0;
                aluOP       = 3;
                OutPortEn   = 0;
                if (R1Le10) begin
                    next_state = S6;
                end else begin
                    next_state = S9;
                end
            end
            S9: begin  // R4 = R4 & R3
                RFSrcMuxSel = 0;
                readAddr1   = 4;
                readAddr2   = 3;
                writeAddr   = 4;
                writeEn     = 1;
                aluOP       = 2;
                OutPortEn   = 0;
                next_state  = S10;
            end
            S10: begin  // R4 = R2 + R3
                RFSrcMuxSel = 0;
                readAddr1   = 2;
                readAddr2   = 3;
                writeAddr   = 4;
                writeEn     = 1;
                aluOP       = 0;
                OutPortEn   = 0;
                next_state  = S11;
            end
            S11: begin  // R4 > R2 
                RFSrcMuxSel = 0;
                readAddr1   = 2;
                readAddr2   = 4;
                writeAddr   = 4;
                writeEn     = 0;
                aluOP       = 0;
                OutPortEn   = 1;
                if (R1Le10) begin
                    next_state = S4;
                end else begin
                    next_state = S12;
                end
            end
            S12: begin  //  OutPort = R4 
                RFSrcMuxSel = 0;
                readAddr1   = 4;
                readAddr2   = 2;
                writeAddr   = 4;
                writeEn     = 0;
                aluOP       = 0;
                OutPortEn   = 1;
                next_state  = HALT;
            end
            HALT: begin  //  HALT
                RFSrcMuxSel = 0;
                readAddr1   = 0;
                readAddr2   = 0;
                writeAddr   = 0;
                writeEn     = 0;
                aluOP       = 0;
                OutPortEn   = 0;
                next_state  = HALT;
            end
        endcase
    end
endmodule
```

## DataPath.sv
```verilog
`timescale 1ns / 1ps

module DataPath (
    input  logic       clk,
    input  logic       reset,
    input  logic       RFSrcMuxSel,
    input  logic [2:0] readAddr1,
    input  logic [2:0] readAddr2,
    input  logic [2:0] writeAddr,
    input  logic       writeEn,
    input  logic       OutPortEn,
    input  logic [1:0] aluOP,
    output logic       R1Le10,
    output logic [7:0] OutPort
);
    logic [7:0] ALUResult, wData;
    logic [7:0] RData1, RData2;

    mux_2x1 U_RFSrcMux (
        .RFSrcMuxSel(RFSrcMuxSel),
        .a          (ALUResult),
        .b          (8'b1),
        .sum        (wData)
    );

    RegFile U_RegFile (
        .clk      (clk),
        .readAddr1(readAddr1),
        .readAddr2(readAddr2),
        .writeAddr(writeAddr),
        .writeEn  (writeEn),
        .wData    (wData),
        .rData1   (RData1),
        .rData2   (RData2)
    );

    comparator U_RiLe10 (
        .rData1(RData1),
        .rData2(RData2),
        .lte   (R1Le10)
    );

    alu U_ALU (
        .rData1(RData1),
        .rData2(RData2),
        .aluOP(aluOP),
        .alu_Data(ALUResult)
    );

    register U_OutPort (
        .clk  (clk),
        .reset(reset),
        .en   (OutPortEn),
        .d    (RData1),
        .q    (OutPort)
    );
endmodule

/////////////////////////////////////////////////////////////////////////////

module mux_2x1 (
    input  logic       RFSrcMuxSel,
    input  logic [7:0] a,
    input  logic [7:0] b,
    output logic [7:0] sum
);
    always_comb begin
        sum = 1'b0;
        case (RFSrcMuxSel)
            0: sum = a;
            1: sum = b;
        endcase
    end
endmodule

module RegFile (
    input  logic       clk,
    input  logic [2:0] readAddr1,
    input  logic [2:0] readAddr2,
    input  logic [2:0] writeAddr,
    input  logic       writeEn,
    input  logic [7:0] wData,
    output logic [7:0] rData1,
    output logic [7:0] rData2
);
    logic [7:0] mem[0:7];

    always_ff @(posedge clk) begin : write
        if (writeEn) begin
            mem[writeAddr] <= wData;
        end
    end

    assign rData1 = (readAddr1 == 3'b0) ? 8'b0 : mem[readAddr1];
    assign rData2 = (readAddr2 == 3'b0) ? 8'b0 : mem[readAddr2];
endmodule

module alu (
    input  logic [7:0] rData1,
    input  logic [7:0] rData2,
    input  logic [1:0] aluOP,
    output logic [7:0] alu_Data
);
    always_comb begin : alu
        alu_Data = rData1 + rData2;
        case (aluOP)
            0: begin
                alu_Data = rData1 + rData2;
            end
            1: begin
                alu_Data = rData1 - rData2;
            end
            2: begin
                alu_Data = rData1 & rData2;
            end
            3: begin
                alu_Data = rData1 | rData2;
            end
        endcase
    end
endmodule

module comparator (
    input logic [7:0] rData1,
    input logic [7:0] rData2,
    output logic lte
);
    assign lte = (rData1 < rData2);
endmodule

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
```