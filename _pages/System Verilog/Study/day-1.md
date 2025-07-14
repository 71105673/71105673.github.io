---
title: "Day-1 Sysyem Verilog" 
date: "2025-07-14"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# SystemVerilog 기본 개념 정리

blocking이고, 시간의 흐름이 없으면? => **Combination Logic**

non-blocking이고, 시간의 흐름이 있으면? => **Sequential Logic**

## `blocking (=)` vs `non-blocking (<=)`

**📌 개념 요약**

| 구분              | `=` (Blocking)                      | `<=` (Non-blocking)                        |
|-------------------|-------------------------------------|--------------------------------------------|
| 이름              | Blocking assignment                | Non-blocking assignment                    |
| 동작 방식         | **순차적 실행** (한 줄씩 차례로 실행됨) | **병렬 실행** (모든 할당이 동시에 예약됨)     |
| 주로 사용 위치    | `combinational logic` (조합 논리)   | `sequential logic` (순차 논리)             |
| 동작 순서         | 즉시 값 할당                         | 이벤트 큐에 값 예약 → 나중에 한 번에 갱신     |
| 시뮬레이션 결과   | 의도치 않은 동작이 발생할 수 있음     | 의도한 동작 유지 가능                      |
| race condition    | 발생 가능성 높음                    | 방지 가능 (순차적 레지스터 설계에 적합)      |


## Latch vs FF

**Latch**
```verilog
module dlatch_rst(rst, clk, d, q);
  input rst, clk, d;
  output reg q;

  always @(*) begin
    if (!rst) q = 1'b0;
    else if (clk) q = d;
  end
endmodule
```

**FF**
```verilog
module d_ff_sync_rst(clk, d, rst_n, q, qb);
  input clk, d, rst_n;
  output reg q;
  output qb;

  assign qb = ~q;

  always @(posedge clk) begin
    if (!rst_n) q <= 1'b0;
    else q <= d;
  end
endmodule
```

**Tip**

- if else륾 많이 쓰면 Power가 증가하여 되도록이면 안쓰는거로!!


## 🧠 SystemVerilog 연산자 정리 모음: Shift & 비교

---

## 🔁 Shift 연산자

| 연산자     | 이름                   | 설명                                                                 |
|------------|------------------------|----------------------------------------------------------------------|
| `<<`       | **논리적 좌측 시프트**   | 비트를 왼쪽으로 이동, 오른쪽은 **0**으로 채움 (부호 무시)             |
| `<<<`      | **산술적 좌측 시프트**   | 부호(sign)를 유지하면서 왼쪽으로 이동 (signed 타입에서 의미 있음)     |
| `>>`       | **논리적 우측 시프트**   | 오른쪽으로 이동, 왼쪽은 **0**으로 채움 (unsigned에 주로 사용)         |
| `>>>`      | **산술적 우측 시프트**   | 부호(sign)를 유지하면서 오른쪽 이동 (signed 변수 전용)  


### 📌 Shift 예시
```verilog
8'b0001_0101 << 2  
 // 결과: 8'b0101_0000
8'sb1001_0101 <<< 2 
// 부호 유지하며 shift (signed만 의미 있음)
```

## 비교 연산자

| 연산자   | 이름            | 설명                                          |
| ----- | ------------- | ------------------------------------------- |
| `==`  | **논리적 동등 비교** | `x`, `z` 같은 미정 상태를 **무시하고 비교** (0 또는 1만 비교) |
| `===` | **4상태 동등 비교** | `x`, `z` 포함한 **완전한 비트 일치 여부**를 비교           |
| `!=`  | 논리적 비교 (다름)   | `x`, `z` 무시하고 다르면 true                      |
| `!==` | 4상태 비교 (다름)   | `x`, `z` 포함하여 완전 불일치 여부 판단                  |

```verilog
4'b010x == 4'b0101   // true → x 무시하고 비교
4'b010x === 4'b0101  // false → x는 무시 불가, bit level mismatch
```









































## 🧪 Testbench 관점에서의 용어

- **Driver**: DUT의 **입력**을 담당  
  → 출력 시에는 `FF(clk)` 처리 필요
- **Monitor**: DUT의 **출력**을 담당  
  → 출력 시에는 **비동기(non-clk)** 처리 필요
- **Sample**: Monitor가 DUT의 출력을 **샘플링**하는 것

---

## 🔧 task vs function

| 항목       | `task`                       | `function`                 |
|------------|------------------------------|----------------------------|
| 시간 흐름   | 있음 (`#`, `@` 사용 가능)       | 없음                        |
| 반환 값     | 없음                         | 있음                        |
| 사용 용도   | 동작 순서나 지연 포함 작업 처리 | 계산, 간단한 로직 등        |

---

## 🗂 파일 확장자

- `.sv`: **SystemVerilog** 파일 → **컴파일러 필요**
- `.v`: **Verilog** 파일

> ✅ `.sv`는 지원 가능한 시뮬레이터나 툴에서만 작동하므로 환경 설정 필요

---

## 🧱 Class 관련 개념

- Class는 **함수형 구조**를 가지며 `task` 사용 가능
- 그러나 내부에서는 `initial`, `always` 사용 불가  
  → 이유: **시간 흐름**이 없기 때문

---

## ⏱ Delay 표현 방법

- `#`: **timescale 기준의 지연**
- `##`: **clock 주기를 지난다는 의미**


```verilog
##4  // clk을 4번 지나는 의미

// 어떤 클럭인지 명확하지 않기 때문에 아래처럼 작성하는 것이 안전:
repeat(4) @(posedge clk);
```

## ⚙️ 기본 데이터 타입
기본 Default 값:

- 값: x
- Signed: unsigned
- 상태: 4-state (0, 1, z, x)

| 타입        | 크기    | Signed | 상태      |
| --------- | ----- | ------ | ------- |
| `integer` | 32bit | Yes    | 4-state |
| `int`     | 32bit | No     | 2-state |

## 📐 Dynamic Array 관련

- 갯수 반드시 선언 필수 → 안 하면 컴파일 에러 발생

- 남는 부분은 X로 초기화됨

- delete() 사용 시 전부 X로 초기화

## Task vs Function

| 항목           | `task`                        | `function`                           |
| ------------ | ----------------------------- | ------------------------------------ |
| **Return 값** | ❌ 없음                          | ✅ 있음 (`return` 필수)                   |
| **시간의 흐름**   | ✅ 가능 (e.g., `#`, `@`)         | ❌ 없음 (지연, 이벤트 사용 불가)                 |
| **호출 위치**    | `initial`, `always` 등 어디서든 가능 | `initial`, `always`, `always_comb` 등 |
| **병렬 처리**    | ✅ 가능 (`fork...join` 등 사용 가능)  | ❌ 불가 (한 번에 하나의 결과만 계산)               |
| **실행 시간**    | 여러 시뮬레이션 사이클 사용 가능            | 한 사이클 내 계산 완료                        |
| **목적**       | 동작, 절차 정의 (ex. 시뮬레이션 환경 제어)   | 계산, 표현식 반환 목적                        |

## ⚡ Glitch (글리치) 정리

### 📌 Glitch란?
**Glitch**는 디지털 회로에서 **의도하지 않은 짧은 신호 변화(펄스)**를 의미합니다.  
특히 **조합 논리 회로**에서 입력이 바뀔 때 출력이 안정되기 전에 잠시 잘못된 값이 나오는 현상입니다.

---

### ⚠️ Glitch 발생 원인

| 원인                          | 설명 |
|-------------------------------|------|
| ⏱️ Propagation Delay 차이     | 입력 간 도달 시간 차이로 출력이 잠시 틀림 |
| 🔗 경로 간 Race Condition     | 여러 논리 경로가 동시에 출력에 영향을 줄 때 발생 |
| 🔁 Feedback Loop에 의한 불안정 | 조합 회로에 피드백이 걸릴 경우 오동작 가능 |


## Svae

**`$save`, `$restart` (Simulator Dependent)**

| 기능         | 설명                                                                |
| ---------- | ----------------------------------------------------------------- |
| `$save`    | 현재 시뮬레이션 상태를 저장 (snapshot 생성)                                     |
| `$restart` | 이전에 저장한 snapshot으로 복귀 (roll back)                                 |
| ⛔ 제한 사항    | 대부분의 commercial 시뮬레이터에서만 지원 (VCS, ModelSim, Questa 등), SV 표준에는 없음 |


**`VCD`, `FSDB` 파일 이용 (waveform 저장)**

| 목적                     | 설명                                        |
| ---------------------- | ----------------------------------------- |
| `$dumpfile`            | 시뮬레이션 중 waveform 파일로 저장 (`.vcd`, `.fsdb`) |
| `$dumpvars`            | 변수 저장 시작                                  |
| `$dumpon` / `$dumpoff` | 특정 타이밍에만 저장 제어 가능                         |
| 복원 여부                  | ❌ 복원은 안 되지만 wave 분석엔 유용                   |



# Verdi & 이용

> x 누르면 시뮬레이션 커서의 값을 확인 가능
 
> 에러 부분이 생겼을 때, 해당 << 버튼을 통해
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-14 145138.png>)
에러 지점 확인 가능
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-14 145203.png>)


> 기본 세팅 Cycle
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-14 150555.png>)

> Error 확인
![text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-14 150853.png>) ![text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-14 151129.png>) ![text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-14 151137.png>)

>save signal
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-14 151410.png>)


## 실습

### shift_register.v
```verilog
`timescale 1ns / 1ps

module shift_reg #(
    parameter WIDTH = 7
)(
    input clk,
    input rstn,
    input signed [WIDTH-1:0] data_in,
    output reg signed [WIDTH-1:0] data_out
);

reg signed [WIDTH-1:0] shift_din [32:0];
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

wire [WIDTH-1:0] shift_dout;
//assign shift_dout = shift_din[32];
assign shift_dout = shift_din[8];

reg [5:0] count;
always @(posedge clk or negedge rstn) begin
    if (~rstn) begin
        count <= 4'b0;
    end
    else begin
        count <= count + 4'b1;
    end
end

reg [WIDTH-1:0] ref_data;
always @(posedge clk or negedge rstn) begin
    if (~rstn) begin
        ref_data <= 4'b0;
    end
    else if (count==6'd1) begin
        ref_data <= data_in;
    end
end

reg [WIDTH-1:0] data_out;
always @(posedge clk or negedge rstn) begin
    if (~rstn) begin
        data_out <= 4'b0;
    end
    else if (count==6'd10) begin
        data_out <= shift_dout;
    end
end

reg shift_op;
always @(posedge clk or negedge rstn) begin
    if (~rstn) begin
        shift_op <= 1'b0;
    end
    else if (count==6'd10) begin
        if (shift_dout == ref_data)
            shift_op <= 1'b0;
        else
            shift_op <= 1'b1;
    end
end

reg error_ind;
always @(posedge clk or negedge rstn) begin
    if (~rstn) begin
        error_ind <= 1'b0;
    end
    else if (count==6'd10) begin
        if (shift_dout == 3)
            error_ind <= 1'b0;
        else
            error_ind <= 1'b1;
    end
end

endmodule
```

### tb_shift_register.v

```verilog
`timescale 1ns/10ps

module tb_shift_reg();

reg clk, rstn;
reg [6:0] data_in;
wire [6:0] data_out;

initial begin
    clk <= 1'b1;
    rstn <= 1'b0;
    #15 rstn <= 1'b1;
    #400 $finish;
end

initial begin
    data_in <= 7'd0;
    //#20 data_in <= 7'd3; // Abnormal
    #25 data_in <= 7'd3; // Normal
    #10 data_in <= 7'd1;
    #10 data_in <= 7'd5;
    #10 data_in <= 7'd11;
    #10 data_in <= 7'd21;
    #10 data_in <= 7'd6;
    #10 data_in <= 7'd8;
    #10 data_in <= 7'd16;
    #10 data_in <= 7'd0;
    #10 data_in <= 7'd3;
    #10 data_in <= 7'd10;
end

shift_reg #( .WIDTH(7) ) i_shift_reg (
    .clk(clk),
    .rstn(rstn),
    .data_in(data_in),
    .data_out(data_out)
);

always #5 clk <= ~clk;

endmodule
```

### run_shift_reg.f
```
tb_shift_register.v
shift_register.v
```

### RUN_shift_register
```
vcs -f run_shift_reg.f -kdb -full64 -debug_access+all+reverse -lca
./simv -verdi &
```