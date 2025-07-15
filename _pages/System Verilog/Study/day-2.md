---
title: "Day-2 Sysyem Verilog" 
date: "2025-07-15"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

## Setup && Hold Time
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 093520.png>)

### 📌 1. Setup Time (설정 시간)

- **정의**: 클럭 상승/하강 에지 **이전**에 데이터 입력(D)이 **안정적으로 유지되어야 하는 최소 시간**.
- **의미**: 클럭 에지가 발생하기 전, 데이터를 미리 준비해야 하는 시간.
- **충족되지 않으면**: 레지스터가 잘못된 데이터를 캡처하거나 메타안정성(Metastability) 문제가 발생할 수 있음.

### 📌 2. Hold Time (유지 시간)

- **정의**: 클럭 상승/하강 에지 **이후**에도 데이터 입력(D)이 **안정적으로 유지되어야 하는 최소 시간**.
- **의미**: 클럭 에지가 발생한 후에도 잠시 동안 데이터를 유지해야 한다는 의미.
- **충족되지 않으면**: 잘못된 데이터가 캡처되거나 메타안정성 발생 가능.

### Recovery / Removal time
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 092225.png>)


### set_clock_transition
- 클럭의 **rise/fall 전이 시간(transition time)** 을 명시적으로 정의하여 **정확한 타이밍 분석**을 수행하도록 합니다.
  
- 클럭 신호가 **0 → 1 또는 1 → 0**으로 바뀌는 데 걸리는 시간 (slew rate) 
### set_max_transition
- 넷(net) 또는 핀(pin)에서 허용되는 최대 전이 시간을 제한합니다.

- 전이 시간이 너무 크면 타이밍 오류 또는 신호 무결성 문제 발생 가능
### set_max_fanout
- 하나의 게이트 출력이 연결 가능한 최대 fanout 수를 제한

- fanout이 많아지면 부하가 커지고, 그에 따라 지연 및 전이 시간 악화됨

### set_input_delay
- **입력 신호**가 클럭 엣지 기준으로 언제 도착하는지를 설정

- 클럭 도메인 간 인터페이스 또는 외부 장치에서 데이터를 받을 때 타이밍 분석에 사용됨
  
### set_output_delay
- 출력 신호가 클럭 엣지 기준으로 언제까지 도착해야 하는지 설정

- 회로가 외부 장치로 데이터를 전송할 때 그 장치의 세팅 타이밍에 맞춰야 함

### set_multicycle_path
- **여러 클럭 사이클(multi-cycle)** 을 사용하는 경로를 명시적으로 설정하여, 타이밍 분석이 과도하게 보수적이지 않도록 조정
- 
### set_false_path
- 실제 회로 동작에서 타이밍을 맞출 필요가 없는 경로를 STA 분석에서 제외시킴

- false path는 실제 데이터 전송이 발생하지 않거나, 타이밍 위반이 발생해도 무관한 경로

***<false_path 예시>*** ⬇️
## 🔀 Clock Domain Crossing (CDC) 

CDC(Clock Domain Crossing)는 서로 다른 클럭 도메인 간에 신호가 전달될 때 발생하는 상황입니다.  

이 때 **메타스테이블 현상(metastability)**, **데이터 손실**, **타이밍 위반** 등이 발생할 수 있기 때문에, **적절한 동기화 및 STA 예외 처리**가 필요합니다.

### ⚠️ 1. 문제점

**📌 Metastability**
- 수신 클럭 도메인에서 신호가 클럭 엣지 근처에 들어오면 레지스터가 불안정한 상태로 유지됨

**📌 데이터 손실**
- 전송 신호가 너무 짧거나 수신 쪽에서 제대로 샘플링하지 못하는 경우

**📌 STA 타이밍 위반**
- 서로 다른 클럭 도메인 간의 경로는 정상적인 타이밍 분석으로 처리할 수 없음

### 🔧 2. 해결 방법

**✅ 동기화 회로(Synchronizer)**
- 단일 비트: 2-stage flip-flop synchronizer 사용
- 멀티 비트: handshake, FIFO 사용

**✅ 타이밍 예외 설정**
CDC 경로는 STA에서 false path 또는 max/min delay로 제약을 걸어줘야 한다.














# 실습 1. RTL Level 

## counter1.v
```verilog
`timescale 1ns/10ps;

module counter1 (
 input clk, rst,
 output [3:0] cnt,
 output ind_cnt
);

reg [3:0] count;

assign cnt = count;

always @(posedge clk or posedge rst) begin
 if (rst) begin
	count <= 4'b0;
 end
 else begin
	count <= count + 4'b1;
 end
end

reg ind_cnt;
always @(posedge clk or posedge rst) begin
	if(rst) begin
		ind_cnt <= 1'b0;
	end
	else if (count == 4'b0010)
		ind_cnt <= 1'b1;
	else
		ind_cnt <= 1'b0;
end


endmodule
```

## counter1_xpor.v
```verilog
`timescale 1ns/10ps;

module counter1_xpro (
 input clk, rst,
 output [3:0] cnt,
 output ind_cnt
);

reg [3:0] count;

assign cnt = count;

//always @(posedge clk or posedge rst) begin
always @(posedge clk) begin
 //if (rst) begin
//	count <= 4'b0;
 //end
 if (count == 4'd15)
	count <= 0;
 else
	count <= count + 4'b1;
end


reg ind_cnt;
always @(posedge clk or posedge rst) begin
	if(rst) begin
		ind_cnt <= 1'b0;
	end
	else if (count == 4'b0010)
		ind_cnt <= 1'b1;
	else
		ind_cnt <= 1'b0;
end

endmodule
```

## counter2.v
```verilog
`timescale 1ns/10ps;

module counter2(
    input clk, rst,
    output [3:0] cnt
);

reg [3:0] count;
assign cnt = count;

always @(posedge clk or posedge rst) begin
	if (rst) begin
        	count <= 4'b0;
   	end
    	else begin
		if(count == 4'd11) begin
			count <= 4'b0;
		end
		else begin
			count <= count + 4'b1;
		end
     	end
end

endmodule
```

## counter3.v
```verilog
`timescale 1ns/10ps;
module counter3(
        input clk, rst,
        output [3:0] cnt1, cnt2
);

reg [3:0] count1, count2;

assign cnt1 = count1;
assign cnt2 = count2;

always @ (posedge clk or posedge rst) begin
        if (rst) begin
                count1 <= 4'b0;
                count2 <= 4'b0;
        end
        else begin
                if (count1==4'd11) begin
                        count1<=4'b0;
                        if (count2==4'd14) begin
                                count2<=4'b0;
                        end
                        else begin
                                count2<=count2+4'b1;
                        end
                end
                else begin
                        count1 <= count1 + 4'b1;
                end
        end
end
endmodule
```

## tb_counter.v
```verilog
`timescale 1ns/10ps;

module tb_cnt();

reg clk, rst;
wire [3:0] cnt1, cnt2, cnt3_1, cnt3_2;
wire ind_cnt1, ind_cnt1_xpro;

initial begin
	clk <= 1'b1;
	rst <= 1'b0;
	#5 rst <=1'b1;
	#5 rst <=1'b0;
	#400 $finish;
end

counter1 TEST1(clk, rst, cnt1, ind_cnt1);
counter1_xpro TEST1_xpro (clk, rst, cnt1_xpro, ind_cnt1_xpro);
counter2 TEST2(clk, rst, cnt2);
counter3 TEST3(clk, rst, cnt3_1, cnt3_2);

always #5 clk <= ~clk;

endmodule
```

## counter_list
```
./counter1.v
./counter1_xpro.v
./counter2.v
./counter3.v
./tb_counter.v
```

## RUN_CNT 
```
vcs -full64 -kdb -debug_access+all+reverse -f counter_list
./simv -verdi &
```

## 결과 -> RTL Simulation
**코드가 잘 돌아가나 시험**
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 115205.png>)





# Synthesis


## counter.list
```
lappend search_path  ../verilog
set net_list "\
../verilog/counter1.v\
"
 analyze -format verilog -library WORK $net_list
```

## run_counter.dc
```
dc_shell -f counter.tcl | tee run.log
```
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 123440.png>)

## counter.tcl
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 124815.png>)


## output/counter1_0/counter1_0.v
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 123557.png>)

## counter.sdc
```verilog
#-----------------------------------------------------------------------
#  case &  clock definition
#-----------------------------------------------------------------------
## FF to FF clock period margin
set CLK_MGN  0.7
## REGIN, REGOUT setup/hold margin
#set io_dly   0.15
set io_dly   0.05

#set per200  "5.00";  # ns -> 200 MHz
#set per5000  "5000.00";  # ps -> 200 MHz
set per1000  "1000.00";  # ps -> 1000 MHz

#set dont_care   "2";
#set min_delay   "0.3";

#set clcon_clk_name "CLK"
#set cnt_clk_period "[expr {$per200*$CLK_MGN}]"
set cnt_clk_period "[expr {$per1000*$CLK_MGN}]"
set cnt_clk_period_h "[expr {$cnt_clk_period/2.0}]"

### I/O DELAY per clock speed
#set cnt_clk_delay         [expr "$per200 * $CLK_MGN * $io_dly"]
set cnt_clk_delay         [expr "$per1000 * $CLK_MGN * $io_dly"]

#-----------------------------------------------------------------------
#  Create  Clock(s)
#-----------------------------------------------------------------------
#create_clock -name clcon_clk     -period [expr "$per875 * $CLK_MGN"] [get_ports {$clcon_clk_name}]
#create_clock -name clcon_clk     -period $clcon_clk_period -waveform "0 $clcon_clk_period_h" [get_ports {$clcon_clk_name}]
create_clock -name cnt_clk       -period $cnt_clk_period   -waveform "0 $cnt_clk_period_h" [get_ports clk]

#LANE 1 RX CLOCK
#create_generated_clock  -name GC_rxck1_org       -source [get_ports I_A_L1_RX_CLKP ] -divide_by 1 [get_pins u_L1_Rswap/U_CM2X1_nand/ZN]
#create_generated_clock  -name GC_rxck1_swp  -add -source [get_ports I_A_L0_RX_CLKP ] -divide_by 1 [get_pins u_L1_Rswap/U_CM2X1_nand/ZN]


#set_clock_uncertainty -setup 0.05 [all_clocks]
set_clock_uncertainty -setup 50 [all_clocks]
#set_clock_uncertainty -hold  0.05 [all_clocks]
set_clock_uncertainty -hold  50 [all_clocks]

# -------------------------------------
#set_driving_cell -no_design_rule -lib_cell BUFFD1BWP35P140 -pin Z  [all_inputs]

set_load            0.2 [all_outputs]
set_max_transition  0.3 [current_design]
set_max_transition  0.15 -clock_path [all_clocks]
set_max_fanout 64       [current_design]
```
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 123943.png>)

## output/counter1_0/counter1_0.timing_max.rpt
```verilog
Information: Updating design information... (UID-85)

****************************************
Report : timing
        -path full
        -delay max
        -max_paths 1
Design : counter1
Version: V-2023.12-SP5-4
Date   : Tue Jul 15 12:31:21 2025
****************************************

Operating Conditions: TT_0P80V_0P00V_0P00V_0P00V_25C   Library: GF22FDX_SC7P5T_116CPP_BASE_CSC20L_TT_0P80V_0P00V_0P00V_0P00V_25C
Wire Load Model Mode: enclosed

  Startpoint: count_reg_0_
              (rising edge-triggered flip-flop clocked by cnt_clk)
  Endpoint: count_reg_2_
            (rising edge-triggered flip-flop clocked by cnt_clk)
  Path Group: cnt_clk
  Path Type: max

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  count_reg_0_/CLK (SC7P5T_SDFFRQX4_CSC20L)               0.00       0.00 r
  count_reg_0_/Q (SC7P5T_SDFFRQX4_CSC20L)                62.79      62.79 f
  U14/Z (SC7P5T_AN2X4_CSC20L)                            15.88      78.68 f
  U13/Z (SC7P5T_ND2X3_CSC20L)                             7.13      85.80 r
  U18/Z (SC7P5T_AOA211X2_CSC20L)                         30.05     115.86 r
  count_reg_2_/D (SC7P5T_SDFFRQX4_CSC20L)                 0.00     115.86 r
  data arrival time                                                115.86

  clock cnt_clk (rise edge)                             700.00     700.00
  clock network delay (ideal)                             0.00     700.00
  clock uncertainty                                     -50.00     650.00
  count_reg_2_/CLK (SC7P5T_SDFFRQX4_CSC20L)               0.00     650.00 r
  library setup time                                    -54.03     595.97
  data required time                                               595.97
  --------------------------------------------------------------------------
  data required time                                               595.97
  data arrival time                                               -115.86
  --------------------------------------------------------------------------
  slack (MET)                                                      480.12
```
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 123753.png>)

## output/counter1_0/counter1_0.timing_min.rpt
**Hole**의 경우는 시간 그래프상 오른쪽이기 때문에 -가 나오면 violation
```verilog

****************************************
Report : timing
        -path full
        -delay min
        -max_paths 1
Design : counter1
Version: V-2023.12-SP5-4
Date   : Tue Jul 15 12:31:21 2025
****************************************

Operating Conditions: TT_0P80V_0P00V_0P00V_0P00V_25C   Library: GF22FDX_SC7P5T_116CPP_BASE_CSC20L_TT_0P80V_0P00V_0P00V_0P00V_25C
Wire Load Model Mode: enclosed

  Startpoint: count_reg_0_
              (rising edge-triggered flip-flop clocked by cnt_clk)
  Endpoint: count_reg_0_
            (rising edge-triggered flip-flop clocked by cnt_clk)
  Path Group: cnt_clk
  Path Type: min

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  count_reg_0_/CLK (SC7P5T_SDFFRQX4_CSC20L)               0.00       0.00 r
  count_reg_0_/Q (SC7P5T_SDFFRQX4_CSC20L)                57.03      57.03 r
  U12/Z (SC7P5T_INVX3_CSC20L)                             8.51      65.54 f
  count_reg_0_/D (SC7P5T_SDFFRQX4_CSC20L)                 0.00      65.54 f
  data arrival time                                                 65.54

  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  clock uncertainty                                      50.00      50.00
  count_reg_0_/CLK (SC7P5T_SDFFRQX4_CSC20L)               0.00      50.00 r
  library hold time                                       3.17      53.17
  data required time                                                53.17
  --------------------------------------------------------------------------
  data required time                                                53.17
  data arrival time                                                -65.54
  --------------------------------------------------------------------------
  slack (MET)                                                       12.37
```
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 125237.png>)


## tb_gate_cnt1.v
```verilog
`timescale 1ps/1fs

module tb_gate_cnt1();

reg clk,rst;
wire [3:0] cnt1;
wire ind_cnt1;

initial begin
 clk <= 1'b1;
 rst <= 1'b0;
 #5 rst <= 1'b1;
 #5 rst <= 1'b0;
 #400 $finish;
end

counter1 GATE_CNT1(clk, rst, cnt1, ind_cnt1);

always #5 clk <= ~clk;

endmodule
```
## gate_cnt_filelist
```
./counter1_0.v
./tb_gate_cnt1.v
```
## run_gate_cnt1
```
vcs -full64 \
    -kdb \
    -debug_access+all \
    -v /pdk/GF22FDX_SC7P5T_116CPP_BASE_CSC20L_FDK_RELV02R80/verilog/GF22FDX_SC7P5T_116CPP_BASE_CSC20L.v \
    -v /pdk/GF22FDX_SC7P5T_116CPP_BASE_CSC20L_FDK_RELV02R80/verilog/prim.v \
    -f gate_cnt_filelist
./simv -verdi &
```

## 결과 -> Pre-Layout Simulation
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 141335.png>)
결과가 동일하게 나오며 이상 없음을 확인






# Gate Level Simulation

## run_counter_xpro.dc
위치를 verilog로 옮긴 후

```
 cp ../syn/output/counter1_xpro_0/counter1_xpro_0.v .
```

![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 145610.png>)


## counter1_xpro_0.v
```verilog
`timescale 1ps/1fs

module counter1_xpro ( clk, rst, cnt, ind_cnt );
  output [3:0] cnt;
  input clk, rst;
  output ind_cnt;
  wire   N6, N7, N8, N9, n3, n4, n5, n60, n70, n80, n90;

  SC7P5T_SDFFQX4_CSC20L count_reg_0_ ( .D(N6), .SI(n4), .SE(n4), .CLK(clk),
        .Q(cnt[0]) );
  SC7P5T_SDFFQX4_CSC20L count_reg_2_ ( .D(N8), .SI(n4), .SE(n4), .CLK(clk),
        .Q(cnt[2]) );
  SC7P5T_SDFFQX4_CSC20L count_reg_1_ ( .D(N7), .SI(n4), .SE(n4), .CLK(clk),
        .Q(cnt[1]) );
  SC7P5T_SDFFQX4_CSC20L count_reg_3_ ( .D(N9), .SI(n4), .SE(n4), .CLK(clk),
        .Q(cnt[3]) );
  SC7P5T_SDFFRQX4_CSC20L ind_cnt_reg ( .D(n80), .SI(n4), .SE(n4), .CLK(clk),
        .RESET(n3), .Q(ind_cnt) );
  SC7P5T_AO22IA1A2X1_CSC20L U12 ( .A1(n60), .A2(cnt[3]), .B1(n60), .B2(cnt[3]),
        .Z(N9) );
  SC7P5T_INVX3_CSC20L U13 ( .A(cnt[0]), .Z(N6) );
  SC7P5T_ND2X4_CSC20L U14 ( .A(n70), .B(cnt[1]), .Z(n60) );
  SC7P5T_OA22IA1A2X2_CSC20L U15 ( .A1(cnt[1]), .A2(cnt[0]), .B1(cnt[0]), .B2(
        cnt[1]), .Z(N7) );
  SC7P5T_AN2X4_CSC20L U16 ( .A(cnt[0]), .B(cnt[2]), .Z(n70) );
  SC7P5T_INVX20_CSC20L U17 ( .A(rst), .Z(n3) );
  SC7P5T_OR3X2_CSC20L U18 ( .A(cnt[2]), .B(n90), .C(cnt[3]), .Z(n5) );
  SC7P5T_INVX2_CSC20L U19 ( .A(n5), .Z(n80) );
  SC7P5T_ND2X2_CSC20L U20 ( .A(N6), .B(cnt[1]), .Z(n90) );
  SC7P5T_AOA211X2_CSC20L U21 ( .C1(cnt[0]), .C2(cnt[1]), .B(cnt[2]), .A(n60),
        .Z(N8) );
  SC7P5T_TIELOX1_CSC20L U22 ( .Z(n4) );
endmodule
```
**더욱 자세한 스케일로 보기 위해 1ps/1fs로 설정**
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 145908.png>)


## tb_gate_cnt1_xpro.v
```verilog
`timescale 1ps/1fs

module tb_gate_cnt1_xpro();

reg clk,rst;
wire [3:0] cnt1;
wire ind_cnt1;

initial begin
 clk <= 1'b1;
 rst <= 1'b0;
 #5 rst <= 1'b1;
 #5 rst <= 1'b0;
 #400 $finish;
end

counter1_xpro GATE_CNT1_XPRO(clk, rst, cnt1, ind_cnt1);

always #5 clk <= ~clk;

endmodule
```

## 결과 -> Gate Level Simulation

> RTL Level에서는 0이 나온 값이 Gate Level에서는 X가 나옴
>
> 코드에서 오류가 나온 이유를 찾아야함
>
> 해당 코드의 경우 X가 뜨기 이전(0ns) 에서부터 오류 발생
>
> tb상의 rst시 0초기화 부분이 없기에 발생한 오류인 것을 확인 가능하다.


![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 152334.png>)















