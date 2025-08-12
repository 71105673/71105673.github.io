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


# Home Work

![alt text](<../../../assets/img/CPU/day_5/화면 캡처 2025-08-12 155354.png>)
![alt text](<../../../assets/img/CPU/day_5/화면 캡처 2025-08-12 155320.png>)
