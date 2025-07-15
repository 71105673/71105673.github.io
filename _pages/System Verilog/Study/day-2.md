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

# 실습

### counter1.v
```verilog
`timescale 1ns/10ps

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

### counter1_xpor.v
```verilog
`timescale 1ns/10ps

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

### counter2.v
```verilog
`timescale 1ns/10ps

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
		if(count == 4'b0011) begin
			count <= 4'b0;
		end
		else begin
			count <= count + 4'b1;
		end
     	end
end

endmodule
```

### counter3.v
```verilog
`timescale 1ns/10ps
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

### tb_counter.v
```verilog
`timescale 1ns/10ps

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

### counter_list
```
./counter1.v
./counter1_xpro.v
./counter2.v
./counter3.v
./tb_counter.v
```

### RUN_CNT
```
vcs -full64 -kdb -debug_access+all+reverse -f counter_list
./simv -verdi &
```