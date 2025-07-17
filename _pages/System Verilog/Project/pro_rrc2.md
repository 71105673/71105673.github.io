---
title: "project RRC-2" 
date: "2025-07-17"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# Synthesis 

## run_rrc_filter.dc
```
dc_shell -f rrc_filter.tcl | tee run.log
```
## rrc_filter.sdc
```verilog
#  case &  clock definition
#-----------------------------------------------------------------------
## FF to FF clock period margin
set CLK_MGN  0.7
## REGIN, REGOUT setup/hold margin
#set io_dly   0.15
set io_dly   0.05


#set per200  "5.00";  # ns -> 200 MHz
#set per5000  "5000.00";  # ps -> 200 MHz
#set per1000  "1000.00";  # ps -> 1000 MHz
set per1250 "800.00";  # ps -> 1250 MHz


#set dont_care   "2";
#set min_delay   "0.3";

#set clcon_clk_name "CLK"
#set cnt_clk_period "[expr {$per200*$CLK_MGN}]"
set cnt_clk_period "[expr {$per1250*$CLK_MGN}]"
set cnt_clk_period_h "[expr {$cnt_clk_period/2.0}]"

### I/O DELAY per clock speed
#set cnt_clk_delay         [expr "$per200 * $CLK_MGN * $io_dly"]
set cnt_clk_delay         [expr "$per1250 * $CLK_MGN * $io_dly"]

```
## rrc_filter.list
```
lappend search_path  ../verilog
set net_list "\
../verilog/rrc_filter.sv\
"
 analyze -format sverilog -library WORK $net_list
```
## run_rrc_filter.dc        
```
dc_shell -f rrc_filter.tcl | tee run.log
```

# synthesis 결과
확인 결과 Setup Violation 발생

## Timing Max -> Setup
```

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  shift_din_reg_15__3_/CLK (SC7P5T_DFFRQX4_S_CSC20L)      0.00       0.00 r
  shift_din_reg_15__3_/Q (SC7P5T_DFFRQX4_S_CSC20L)       50.61      50.61 r
  U999/Z (SC7P5T_INVX6_CSC20L)                           10.13      60.75 f
  U1035/S (SC7P5T_FAX6_CSC20L)                           52.11     112.86 r
  U683/Z (SC7P5T_XNR2X2_CSC20L)                          36.98     149.84 f
  U682/Z (SC7P5T_XNR2X2_CSC20L)                          34.38     184.23 r
  U736/CO (SC7P5T_FAX2_A_CSC20L)                         38.24     222.47 r
  U758/S (SC7P5T_FAX2_A_CSC20L)                          57.55     280.02 f
  U846/CO (SC7P5T_FAX2_A_CSC20L)                         35.60     315.62 f
  U910/S (SC7P5T_FAX2_A_CSC20L)                          53.26     368.88 r
  U255/Z (SC7P5T_XOR3X2_CSC20L)                          42.58     411.46 f
  U494/Z (SC7P5T_NR2X4_CSC20L)                           14.97     426.43 r
  U160/Z (SC7P5T_NR2X3_CSC20L)                            9.68     436.11 f
  U115/Z (SC7P5T_INVX3_CSC20L)                            7.99     444.10 r
  U256/Z (SC7P5T_NR2IAX2_CSC20L)                          7.34     451.45 f
  U237/Z (SC7P5T_NR2X2_MR_CSC20L)                        14.95     466.40 r
  U253/Z (SC7P5T_OAI21X2_CSC20L)                         15.18     481.58 f
  U239/Z (SC7P5T_XNR2X1_CSC20L)                          23.06     504.65 f
  U739/Z (SC7P5T_OAI21X2_CSC20L)                         11.82     516.47 r
  data_out_reg_4_/D (SC7P5T_SDFFRQX2_A_CSC20L)            0.00     516.47 r
  data arrival time                                                516.47

  clock cnt_clk (rise edge)                             560.00     560.00
  clock network delay (ideal)                             0.00     560.00
  clock uncertainty                                     -50.00     510.00
  data_out_reg_4_/CLK (SC7P5T_SDFFRQX2_A_CSC20L)          0.00     510.00 r
  library setup time                                    -39.62     470.38
  data required time                                               470.38
  --------------------------------------------------------------------------
  data required time                                               470.38
  data arrival time                                               -516.47
  --------------------------------------------------------------------------
  slack (VIOLATED)                                                 -46.09
```
![alt text](<../../../assets/img/SystemVerilog/rrc2/스크린샷 2025-07-17 093211.png>)

## Timing Min -> Hold
```
  Path Type: min

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  shift_din_reg_5__6_/CLK (SC7P5T_DFFRQX4_S_CSC20L)       0.00       0.00 r
  shift_din_reg_5__6_/Q (SC7P5T_DFFRQX4_S_CSC20L)        48.48      48.48 f
  shift_din_reg_6__6_/D (SC7P5T_DFFRQX1_AS_CSC20L)        0.00      48.48 f
  data arrival time                                                 48.48

  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  clock uncertainty                                      50.00      50.00
  shift_din_reg_6__6_/CLK (SC7P5T_DFFRQX1_AS_CSC20L)      0.00      50.00 r
  library hold time                                      12.95      62.95
  data required time                                                62.95
  --------------------------------------------------------------------------
  data required time                                                62.95
  data arrival time                                                -48.48
  --------------------------------------------------------------------------
  slack (VIOLATED)                                                 -14.47
```
![alt text](<../../../assets/img/SystemVerilog/rrc2/스크린샷 2025-07-17 093723.png>)


## 고찰
Logic Delay가 너무 길어 생긴 상황
따라서 Pipe Path를 넣어서 Setup Delay를 해결하고자 함


# Pipe 구조 변경 synthesis
```verilog
`timescale 1ns / 1ps

module rrc_filter #(
    parameter WIDTH = 7
)(
    input         clk,
    input         rstn,

    input  [WIDTH-1:0]        data_in,   // format : <1.6>
    output logic signed [WIDTH-1:0] data_out
);


// format <8.14> -> 16bit 표현하기 위한 범위 설정
logic signed [WIDTH+9-1:0] mul_00, mul_01, mul_02, mul_03;
logic signed [WIDTH+9-1:0] mul_04, mul_05, mul_06, mul_07;
logic signed [WIDTH+9-1:0] mul_08, mul_09, mul_10, mul_11;
logic signed [WIDTH+9-1:0] mul_12, mul_13, mul_14, mul_15;
logic signed [WIDTH+9-1:0] mul_16, mul_17, mul_18, mul_19;
logic signed [WIDTH+9-1:0] mul_20, mul_21, mul_22, mul_23;
logic signed [WIDTH+9-1:0] mul_24, mul_25, mul_26, mul_27;
logic signed [WIDTH+9-1:0] mul_28, mul_29, mul_30, mul_31;
logic signed [WIDTH+9-1:0] mul_32;

logic signed [WIDTH-1:0] shift_din [32:0];
integer i;
always@(posedge clk or negedge rstn) begin
    if (~rstn) begin
        for(i = 32; i >= 0; i=i-1) begin
            shift_din[i] <= 0;
        end
    end
    else begin
        for(i = 32; i > 0; i=i-1) begin
            shift_din[i] <= shift_din[i-1];
        end
        shift_din[0] <= data_in;
    end
end

// format : <1.8> 에 해당되는 coeff 곱하여 가중치 계산
// <1.6> din * <1.8> coeff -> <2.14> format
always_ff @(posedge clk or negedge rstn) begin
	if(~rstn) begin
        mul_00 <= 'h0;
        mul_01 <= 'h0;
        mul_02 <= 'h0;
        mul_03 <= 'h0;
        mul_04 <= 'h0;
        mul_05 <= 'h0;
        mul_06 <= 'h0;
        mul_07 <= 'h0;
        mul_08 <= 'h0;
        mul_09 <= 'h0;
        mul_10 <= 'h0;
        mul_11 <= 'h0;
        mul_12 <= 'h0;
        mul_13 <= 'h0;
        mul_14 <= 'h0;
        mul_15 <= 'h0;
        mul_16 <= 'h0;
        mul_17 <= 'h0;
        mul_18 <= 'h0;
        mul_19 <= 'h0;
        mul_20 <= 'h0;
        mul_21 <= 'h0;
        mul_22 <= 'h0;
        mul_23 <= 'h0;
        mul_24 <= 'h0;
        mul_25 <= 'h0;
        mul_26 <= 'h0;
        mul_27 <= 'h0;
        mul_28 <= 'h0;
        mul_29 <= 'h0;
        mul_30 <= 'h0;
        mul_31 <= 'h0;
        mul_32 <= 'h0;
    end
    else begin
        mul_00 <= shift_din[00]*0;
        mul_01 <= shift_din[01]*-1;
        mul_02 <= shift_din[02]*1;
        mul_03 <= shift_din[03]*0;
        mul_04 <= shift_din[04]*-1;
        mul_05 <= shift_din[05]*2;
        mul_06 <= shift_din[06]*0;
        mul_07 <= shift_din[07]*-2;
        mul_08 <= shift_din[08]*2;
        mul_09 <= shift_din[09]*0;
        mul_10 <= shift_din[10]*-6;
        mul_11 <= shift_din[11]*8;
        mul_12 <= shift_din[12]*10;
        mul_13 <= shift_din[13]*-28;
        mul_14 <= shift_din[14]*-14;
        mul_15 <= shift_din[15]*111;
        mul_16 <= shift_din[16]*196;
        mul_17 <= shift_din[17]*111;
        mul_18 <= shift_din[18]*-14;
        mul_19 <= shift_din[19]*-28;
        mul_20 <= shift_din[20]*10;
        mul_21 <= shift_din[21]*8;
        mul_22 <= shift_din[22]*-6;
        mul_23 <= shift_din[23]*0;
        mul_24 <= shift_din[24]*2;
        mul_25 <= shift_din[25]*-2;
        mul_26 <= shift_din[26]*0;
        mul_27 <= shift_din[27]*2;
        mul_28 <= shift_din[28]*-1;
        mul_29 <= shift_din[29]*0;
        mul_30 <= shift_din[30]*1;
        mul_31 <= shift_din[31]*-1;
        mul_32 <= shift_din[32]*0;
    end
end

logic signed [WIDTH+16-1:0] filter_sum_1;
logic signed [WIDTH+16-1:0] filter_sum_2;

//always_comb begin
// <2.14> format -> 33개 그럼 여유롭게 32<33<64 
// <2.14> 를 64개 더하면 -> <8.14> 2+ 2^6 -> 2+8 = 8
always @(*) begin
    filter_sum_1 <= mul_00 + mul_01 + mul_02 + mul_03 +
                 mul_04 + mul_05 + mul_06 + mul_07 +
                 mul_08 + mul_09 + mul_10 + mul_11 +
                 mul_12 + mul_13 + mul_14 + mul_15;

    filter_sum_2 <= mul_16 + mul_17 + mul_18 + mul_19 +
                 mul_20 + mul_21 + mul_22 + mul_23 +
                 mul_24 + mul_25 + mul_26 + mul_27 +
                 mul_28 + mul_29 + mul_30 + mul_31 +
                 mul_32;
end

logic signed [WIDTH+16-1:0] filter_sum;
assign filter_sum = filter_sum_1 + filter_sum_2;

// Truncation  <8.14> 22bit를 뒷자리 8만큼 짤라서 (<1.6>으로 만들기 위해 소수보면 14 - 6 = 8)
logic signed [WIDTH+8-1:0] trunc_filter_sum;
assign trunc_filter_sum = filter_sum[WIDTH+16-1:8];

// Saturation <1.6> 최종 출력을 위해 (7bit) 범위 -64~
always_ff @(posedge clk or negedge rstn) begin
    if (~rstn) begin
        data_out <= 'h0;
    end 
    else if (trunc_filter_sum >= 63)begin
        data_out <= 63;
    end
    else if (trunc_filter_sum < -64)begin
        data_out <= -64;
    end
    else begin
        data_out <= trunc_filter_sum[WIDTH-1:0];
    end
end

endmodule
```

## Verdi
![alt text](<../../../assets/img/SystemVerilog/rrc2/스크린샷 2025-07-17 110558.png>)


# Synthesis 결과

## Timing Max -> Setup
```

  Path Type: max

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  mul_28_reg_10_/CLK (SC7P5T_SDFFRQNX4_CSC20L)            0.00       0.00 r
  mul_28_reg_10_/QN (SC7P5T_SDFFRQNX4_CSC20L)            46.04      46.04 f
  U1042/S (SC7P5T_FAX2_A_CSC20L)                         60.86     106.90 r
  U1044/CO (SC7P5T_FAX2_A_CSC20L)                        37.49     144.39 r
  U715/Z (SC7P5T_XOR3X2_CSC20L)                          66.89     211.28 f
  U793/S (SC7P5T_FAX2_A_CSC20L)                          56.68     267.96 r
  U293/Z (SC7P5T_XOR3X2_CSC20L)                          67.46     335.42 f
  U276/Z (SC7P5T_XNR2X2_CSC20L)                          31.58     366.99 f
  U210/Z (SC7P5T_NR2X4_CSC20L)                           16.34     383.33 r
  U223/Z (SC7P5T_OAI21X4_CSC20L)                         15.36     398.69 f
  U229/Z (SC7P5T_AOI21X6_CSC20L)                         14.95     413.65 r
  U652/Z (SC7P5T_INVX4_CSC20L)                            9.13     422.77 f
  U316/Z (SC7P5T_AOI21X2_CSC20L)                         12.50     435.27 r
  U245/Z (SC7P5T_XNR2X1_CSC20L)                          27.47     462.75 f
  U1096/Z (SC7P5T_OAI21X2_CSC20L)                        11.82     474.57 r
  data_out_reg_4_/D (SC7P5T_SDFFRQX2_A_CSC20L)            0.00     474.57 r
  data arrival time                                                474.57

  clock cnt_clk (rise edge)                             560.00     560.00
  clock network delay (ideal)                             0.00     560.00
  clock uncertainty                                     -50.00     510.00
  data_out_reg_4_/CLK (SC7P5T_SDFFRQX2_A_CSC20L)          0.00     510.00 r
  library setup time                                    -39.68     470.32
  data required time                                               470.32
  --------------------------------------------------------------------------
  data required time                                               470.32
  data arrival time                                               -474.57
  --------------------------------------------------------------------------
  slack (VIOLATED)                                                  -4.25
```
![alt text](<../../../assets/img/SystemVerilog/rrc2/스크린샷 2025-07-17 110827.png>)

## Timing Min -> Hold
```

  Path Type: min

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  shift_din_reg_23__3_/CLK (SC7P5T_DFFRQX4_S_CSC20L)      0.00       0.00 r
  shift_din_reg_23__3_/Q (SC7P5T_DFFRQX4_S_CSC20L)       47.04      47.04 f
  shift_din_reg_24__3_/D (SC7P5T_DFFRQX1_AS_CSC20L)       0.00      47.04 f
  data arrival time                                                 47.04

  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  clock uncertainty                                      50.00      50.00
  shift_din_reg_24__3_/CLK (SC7P5T_DFFRQX1_AS_CSC20L)     0.00      50.00 r
  library hold time                                      13.32      63.32
  data required time                                                63.32
  --------------------------------------------------------------------------
  data required time                                                63.32
  data arrival time                                                -47.04
  --------------------------------------------------------------------------
  slack (VIOLATED)                                                 -16.27
```
![alt text](<../../../assets/img/SystemVerilog/rrc2/스크린샷 2025-07-17 110931.png>)

## 고찰
아직 Setup Violation이 발생 그러나 이전에 비해 많은 개선이 일어남 추가 수정을 통해 오류를 잡기위해 stage를 더 나누기 시도

# Pipe 추가 
**Sum 부분의 계산에 FF 추가**
```verilog

logic signed [WIDTH+16-1:0] filter_sum_reg;

always_ff @(posedge clk or negedge rstn) begin
        if(~rstn) begin
                filter_sum_reg <= 'h0;
        end
        else begin
                filter_sum_reg <= filter_sum_1 + filter_sum_2;
        end
end

// Truncation  <8.14> 22bit를 뒷자리 8만큼 짤라서 (<1.6>으로 만들기 위해 소수보면 14 - 6 = 8)
logic signed [WIDTH+8-1:0] trunc_filter_sum;
assign trunc_filter_sum = filter_sum_reg[WIDTH+16-1:8];

// Saturation <1.6> 최종 출력을 위해 (7bit) 범위 -64~
always_ff @(posedge clk or negedge rstn) begin
    if (~rstn) begin
        data_out <= 'h0;
    end
    else if (trunc_filter_sum >= 63)begin
        data_out <= 63;
    end
    else if (trunc_filter_sum < -64)begin
        data_out <= -64;
    end
    else begin
        data_out <= trunc_filter_sum[WIDTH-1:0];
    end
end

endmodule
```
# Synthesis
## Timing Max -> Setup
```

  Path Type: max

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  mul_19_reg_6_/CLK (SC7P5T_SDFFRQX4_S_CSC20L)            0.00       0.00 r
  mul_19_reg_6_/Q (SC7P5T_SDFFRQX4_S_CSC20L)             50.30      50.30 f
  U780/S (SC7P5T_FAX2_A_CSC20L)                          58.68     108.97 r
  U812/S (SC7P5T_FAX2_A_CSC20L)                          56.77     165.74 f
  U535/Z (SC7P5T_XNR2X2_CSC20L)                          39.25     204.99 r
  U534/Z (SC7P5T_XNR2X2_CSC20L)                          29.53     234.52 f
  U277/Z (SC7P5T_OA21X4_CSC20L)                          23.39     257.92 f
  U276/Z (SC7P5T_NR2X4_CSC20L)                           11.41     269.32 r
  U532/Z (SC7P5T_NR2IAX4_CSC20L)                         18.41     287.74 r
  U400/Z (SC7P5T_OAI21X4_CSC20L)                         12.96     300.70 f
  U518/Z (SC7P5T_XNR2X2_CSC20L)                          38.40     339.10 r
  U517/Z (SC7P5T_XNR2X2_CSC20L)                          30.14     369.24 f
  U218/Z (SC7P5T_NR2X4_CSC20L)                           11.79     381.04 r
  U215/Z (SC7P5T_NR2X2_MR_CSC20L)                        10.29     391.33 f
  U214/Z (SC7P5T_NR2X4_CSC20L)                           12.34     403.67 r
  U230/Z (SC7P5T_ND2X6_CSC20L)                           14.91     418.58 f
  U232/Z (SC7P5T_AO21IAX2_CSC20L)                        24.40     442.98 f
  U551/Z (SC7P5T_XOR2X2_CSC20L)                          27.76     470.74 r
  filter_sum_reg_reg_16_/D (SC7P5T_SDFFRQX2_A_CSC20L)     0.00     470.74 r
  data arrival time                                                470.74

  clock cnt_clk (rise edge)                             560.00     560.00
  clock network delay (ideal)                             0.00     560.00
  clock uncertainty                                     -50.00     510.00
  filter_sum_reg_reg_16_/CLK (SC7P5T_SDFFRQX2_A_CSC20L)
                                                          0.00     510.00 r
  library setup time                                    -39.12     470.88
  data required time                                               470.88
  --------------------------------------------------------------------------
  data required time                                               470.88
  data arrival time                                               -470.74
  --------------------------------------------------------------------------
  slack (MET)                                                        0.14
```
![alt text](<../../../assets/img/SystemVerilog/rrc2/스크린샷 2025-07-17 112043.png>)

## Timing Min -> Hold
```

  Path Type: min

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  shift_din_reg_23__0_/CLK (SC7P5T_DFFRQX4_S_CSC20L)      0.00       0.00 r
  shift_din_reg_23__0_/Q (SC7P5T_DFFRQX4_S_CSC20L)       47.04      47.04 f
  shift_din_reg_24__0_/D (SC7P5T_DFFRQX1_AS_CSC20L)       0.00      47.04 f
  data arrival time                                                 47.04

  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  clock uncertainty                                      50.00      50.00
  shift_din_reg_24__0_/CLK (SC7P5T_DFFRQX1_AS_CSC20L)     0.00      50.00 r
  library hold time                                      13.32      63.32
  data required time                                                63.32
  --------------------------------------------------------------------------
  data required time                                                63.32
  data arrival time                                                -47.04
  --------------------------------------------------------------------------
  slack (VIOLATED)                                                 -16.27
```
![alt text](<../../../assets/img/SystemVerilog/rrc2/스크린샷 2025-07-17 112105.png>)

## 추가 실험
```verilog
logic signed [WIDTH+16-1:0] filter_sum_1;
logic signed [WIDTH+16-1:0] filter_sum_2;
logic signed [WIDTH+16-1:0] filter_sum_3;
logic signed [WIDTH+16-1:0] filter_sum_4;

//always_comb begin
// <2.14> format -> 33개 그럼 여유롭게 32<33<64 
// <2.14> 를 64개 더하면 -> <8.14> 2+ 2^6 -> 2+8 = 8
always @(*) begin
    filter_sum_1 <= mul_00 + mul_01 + mul_02 + mul_03 +
                 mul_04 + mul_05 + mul_06 + mul_07;

    filter_sum_2 <= mul_08 + mul_09 + mul_10 + mul_11 +
                 mul_12 + mul_13 + mul_14 + mul_15;

    filter_sum_3 <= mul_16 + mul_17 + mul_18 + mul_19 +
                 mul_20 + mul_21 + mul_22 + mul_23;

    filter_sum_4 <= mul_24 + mul_25 + mul_26 + mul_27 +
                 mul_28 + mul_29 + mul_30 + mul_31 +
                 mul_32;
end

logic signed [WIDTH+16-1:0] filter_sum_reg;

always_ff @(posedge clk or negedge rstn) begin
	if(~rstn) begin
		filter_sum_reg <= 'h0;
	end
	else begin
		filter_sum_reg <= filter_sum_1 + filter_sum_2 + filter_sum_3 + filter_sum_4;
	end
end
```

## Timing Max -> Setup
```

  Path Type: max

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  mul_13_reg_7_/CLK (SC7P5T_SDFFRQX4_CSC20L)              0.00       0.00 r
  mul_13_reg_7_/Q (SC7P5T_SDFFRQX4_CSC20L)               59.32      59.32 f
  U804/S (SC7P5T_FAX2_A_CSC20L)                          54.36     113.67 r
  U812/S (SC7P5T_FAX2_A_CSC20L)                          57.70     171.38 f
  U818/S (SC7P5T_FAX2_A_CSC20L)                          61.07     232.45 r
  U541/Z (SC7P5T_XNR2X2_CSC20L)                          38.87     271.32 f
  U528/Z (SC7P5T_XNR2X2_CSC20L)                          31.62     302.94 r
  U251/Z (SC7P5T_XNR2X2_CSC20L)                          36.30     339.24 f
  U245/Z (SC7P5T_XNR2X2_CSC20L)                          34.62     373.86 r
  U274/Z (SC7P5T_ND2IAX2_CSC20L)                         13.42     387.29 f
  U292/Z (SC7P5T_OA21X4_CSC20L)                          24.55     411.84 f
  U236/Z (SC7P5T_OAI21X6_CSC20L)                         12.13     423.97 r
  U220/Z (SC7P5T_INVX2_CSC20L)                           11.33     435.30 f
  U542/Z (SC7P5T_XNR2X2_CSC20L)                          29.94     465.24 r
  filter_sum_reg_reg_9_/D (SC7P5T_SDFFRQX4_S_CSC20L)      0.00     465.24 r
  data arrival time                                                465.24

  clock cnt_clk (rise edge)                             560.00     560.00
  clock network delay (ideal)                             0.00     560.00
  clock uncertainty                                     -50.00     510.00
  filter_sum_reg_reg_9_/CLK (SC7P5T_SDFFRQX4_S_CSC20L)
                                                          0.00     510.00 r
  library setup time                                    -44.26     465.74
  data required time                                               465.74
  --------------------------------------------------------------------------
  data required time                                               465.74
  data arrival time                                               -465.24
  --------------------------------------------------------------------------
  slack (MET)                                                        0.50
```

## Timing Min -> Hold
```

  Path Type: min

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  shift_din_reg_23__2_/CLK (SC7P5T_DFFRQX4_S_CSC20L)      0.00       0.00 r
  shift_din_reg_23__2_/Q (SC7P5T_DFFRQX4_S_CSC20L)       47.04      47.04 f
  shift_din_reg_24__2_/D (SC7P5T_DFFRQX1_AS_CSC20L)       0.00      47.04 f
  data arrival time                                                 47.04

  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  clock uncertainty                                      50.00      50.00
  shift_din_reg_24__2_/CLK (SC7P5T_DFFRQX1_AS_CSC20L)     0.00      50.00 r
  library hold time                                      13.32      63.32
  data required time                                                63.32
  --------------------------------------------------------------------------
  data required time                                                63.32
  data arrival time                                                -47.04
  --------------------------------------------------------------------------
  slack (VIOLATED)                                                 -16.27
```