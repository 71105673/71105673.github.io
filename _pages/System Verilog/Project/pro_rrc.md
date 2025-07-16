---
title: "project RRC" 
date: "2025-07-16"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# 과제 기본 개념 정리
![text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-15 161918.png>) 
![text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-15 162201.png>) 
![text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-15 162632.png>) 
![text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-15 163210.png>)
- 고정소숫점 방식 -> 비트수가 작으면 노이즈가 증가, 그러나 area는 작음
 
- 이것을 fixed point simulation

![alt text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-15 163719.png>)
- RRC Filter -> 주파수 도메인 확인

- 여기서 fixed = 1.8인 이유가 정수부가 작아서

- 즉 정수부 1, 소수부 8 = 9bit

# 프로젝트 
> data = <1.6>, coeff = <1.8>
> 
> 즉, 7 x 9 = 16bit가 필요하다. / <2.14>

>**2Tap 기준 예시**
> <2.14> + <2.14> = <3.14> 
>
>**16Tap 이면??**
> <2.14> + <2.14> + ... = <6.14>
>
>**64Tap 이면??** 
> <2.14> + <2.14> + ... = <8.14>

만약 <1.6>을 <2.5> format으로 하면 Gain이 작아짐

마지막에 <8.14> => <1.6>으로 해야함

상위 7bit -> saturation

하위 7bit -> Truncation

## 🚫 Saturation (상위 비트 포화 처리)

- 정수부 8비트를 1비트로 줄이면 범위 초과가 발생할 수 있음
- **범위 초과 시**, 최대/최소값으로 고정

| 원래 값 (8.14 기준)         | 변환 결과 (1.6 기준)       | 설명                |
|-----------------------------|-----------------------------|---------------------|
| 00000001.xxxxxxxxxxxxxx     | 그대로 유지                | 표현 가능           |
| 01111111.xxxxxxxxxxxxxx     | 1.111111                    | 양의 최대값으로 포화 |
| 10000000.xxxxxxxxxxxxxx     | -1.000000                   | 음의 최소값으로 포화 |


## ✂️ Truncation (절삭)

- **정의**: 고정소수점 수에서 표현 비트 수를 줄일 때, 하위 비트(덜 중요한 비트)를 단순히 잘라내는 방법.
- **목적**: 소수점 자리수를 줄여서 비트 수를 맞추기 위해 사용.
- **특징**: 버림 방식이기 때문에 약간의 값 손실(precision loss)이 발생할 수 있음.

---


# 코드

## rrc._filter.sv
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
always @(*) begin
    mul_00 = shift_din[00]*0;
    mul_01 = shift_din[01]*-1;
    mul_02 = shift_din[02]*1;
    mul_03 = shift_din[03]*0;
    mul_04 = shift_din[04]*-1;
    mul_05 = shift_din[05]*2;
    mul_06 = shift_din[06]*0;
    mul_07 = shift_din[07]*-2;
    mul_08 = shift_din[08]*2;
    mul_09 = shift_din[09]*0;
    mul_10 = shift_din[10]*-6;
    mul_11 = shift_din[11]*8;
    mul_12 = shift_din[12]*10;
    mul_13 = shift_din[13]*-28;
    mul_14 = shift_din[14]*-14;
    mul_15 = shift_din[15]*111;
    mul_16 = shift_din[16]*196;
    mul_17 = shift_din[17]*111;
    mul_18 = shift_din[18]*-14;
    mul_19 = shift_din[19]*-28;
    mul_20 = shift_din[20]*10;
    mul_21 = shift_din[21]*8;
    mul_22 = shift_din[22]*-6;
    mul_23 = shift_din[23]*0;
    mul_24 = shift_din[24]*2;
    mul_25 = shift_din[25]*-2;
    mul_26 = shift_din[26]*0;
    mul_27 = shift_din[27]*2;
    mul_28 = shift_din[28]*-1;
    mul_29 = shift_din[29]*0;
    mul_30 = shift_din[30]*1;
    mul_31 = shift_din[31]*-1;
    mul_32 = shift_din[32]*0;
end

logic signed [WIDTH+16-1:0] filter_sum;

//always_comb begin
// <2.14> format -> 33개 그럼 여유롭게 32<33<64 
// <2.14> 를 64개 더하면 -> <8.14> 2+ 2^6 -> 2+8 = 8
always @(*) begin
    filter_sum = mul_00 + mul_01 + mul_02 + mul_03 +
                 mul_04 + mul_05 + mul_06 + mul_07 +
                 mul_08 + mul_09 + mul_10 + mul_11 +
                 mul_12 + mul_13 + mul_14 + mul_15 +
                 mul_16 + mul_17 + mul_18 + mul_19 +
                 mul_20 + mul_21 + mul_22 + mul_23 +
                 mul_24 + mul_25 + mul_26 + mul_27 +
                 mul_28 + mul_29 + mul_30 + mul_31 +
                 mul_32;
end

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

## tb_rrc_filter.sv
```verilog
`timescale 1ns/10ps

module tb_rrc_filter();

logic clk, rstn;
logic signed [6:0] data_in;
logic signed [6:0] data_out;
logic signed [6:0] adc_data_in [0:93695];

initial begin
    clk <= 1'b1;
    rstn <= 1'b0;
    #55 rstn <= 1'b1;
    // #500000 $finish;
end

always #5 clk <= ~clk;

integer fd_adc_di;
integer fd_rrc_do;
integer i;
int data;
initial begin
    //always @(posedge clk) begin
    fd_adc_di = $fopen("./ofdm_i_adc_serial_out_fixed_30dB.txt", "r");
    //fd_adc_di = $fopen("./ofdm_adc_serial_out_fixed_30dB.txt", "r");
    fd_rrc_do = $fopen("./rrc_do.txt", "w");
    i = 0;
    while (!$feof(fd_adc_di)) begin
        void'($fscanf(fd_adc_di,"%d\n",data));
        adc_data_in[i] = data;
        i = i + 1;
    end
    #800000 $finish;
    $fclose(fd_rrc_do);
end

logic [23:0] adc_dcnt;
always_ff @(posedge clk or negedge rstn) begin
    if (~rstn)
        adc_dcnt <= 'h0;
    else
        adc_dcnt <= adc_dcnt + 1'b1;
end

logic [6:0] tmp_data_in;
assign tmp_data_in = adc_data_in[adc_dcnt];
always_ff @(posedge clk or negedge rstn) begin
    if (~rstn)
        data_in <= 'h0;
    else
        data_in <= tmp_data_in;
end

always_ff @(negedge clk) begin
    //$fd_rrc_do = $fopen("./rrc_do.txt", "w");
    $fwrite(fd_rrc_do, "%0d\n", data_out);
end

rrc_filter #(.WIDTH(7)) i_rrc_filter (
    .clk(clk),
    .rstn(rstn),
    .data_in(data_in),
    .data_out(data_out)
);

endmodule
```

## rrc_filelist
```
./rrc_filter.sv
./tb_rrc_filter.sv
```

## run_rrc_filter2 -> verdi 실행용
```
vcs -full64 -sverilog -kdb -debug_access+all+reverse -f rrc_filelist
./simv -verdi &
```

## 결과
![alt text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-16 105522.png>)

> Truncation에서 하위 8bit를 자른 후, 
>
> Saturation이 되며 상위 비트를 처리하는 것을 확인할 수 있다.

## run_rrc_filter -> txt 파일 생성
```
vcs -sverilog -full64 -debug_all \
	rrc_filter.sv tb_rrc_filter.sv \
	-o simv && ./simv
```

해당 코드를 통해 rrc_do.txt를 출력하여 MATLAB을 통해 파형 확인

## MATLAB 결과
```matlab
>> % 데이터 불러오기
y1 = load('rrc_do_pro.txt');
y2 = load('my_rrc_do.txt');

% 서브플롯: 좌우 1행 2열
subplot(1, 2, 1);            % 왼쪽 그래프
plot(y1);
xlabel('Sample');
ylabel('Amplitude');
title('rrc\_do\_pro.txt');
grid on;

subplot(1, 2, 2);            % 오른쪽 그래프
plot(y2);
xlabel('Sample');
ylabel('Amplitude');
title('my\_rrc\_do.txt');
grid on;
```
![alt text](<../../../assets/img/SystemVerilog/rrc/스크린샷 2025-07-16 110058.png>)

제시된 결과와 비슷한 파형을 확인 가능