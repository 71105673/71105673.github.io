---
title: "System Verolog Day-1"
date: "2025-08-06"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---
# 버튼을 통해 Up / Down 모드를 수정하는 10000진 카운터
<video controls src="../../../assets/img/CPU/0806자막.mp4" title="Title"></video>

# Top Module
```verilog
`timescale 1ns / 1ps

module top_UpDownCounter(
    input logic clk,
    input logic reset,
    input logic btnU,
    output logic [3:0] fndCom,
    output logic [7:0] fndFont
);
    logic [13:0] count;
    logic U_btn_db;

    btn_debounce UBTN_DB(
        .clk(clk),
        .rst(reset),
        .i_btn(btnU),
        .o_btn(U_btn_db)
    );

    UpDownCounter U_UD_COUNNTER(
        .clk(clk),
        .reset(reset),
        .btnU(U_btn_db),
        .count(count)
    );

    fnd_Controller U_FND_CNT(
        .clk(clk),
        .reset(reset),
        .number(count),
        .fndCom(fndCom),
        .fndFont(fndFont)
    );
endmodule
```

## Up Down counter
```verilog
`timescale 1ns / 1ps

module UpDownCounter (
    input  logic        clk,
    input  logic        reset,
    input  logic        btnU,
    output logic [13:0] count
);

    logic tick_10hz;
    
    logic mode;

    clk_div_10hz U_Clk_Dib_10hz (
        .clk  (clk),
        .reset(reset),
        .tick_10hz(tick_10hz)
    );

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            mode <= 0;
        end else begin
            if (btnU) begin 
                mode <= ~mode;
            end
        end
    end

    up_down_counter U_Up_Down_counter (
        .clk  (clk),
        .reset(reset),
        .tick (tick_10hz),
        .mode (mode),
        .count(count)
    );

endmodule

////////////////////////////////////////////////////////////////////////////////////

module up_down_counter (
    input  logic        clk,
    input  logic        reset,
    input  logic        tick,
    input  logic        mode,
    output logic [13:0] count
);
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            count <= 0;
        end else begin
            if (!mode) begin  // up
                if (tick) begin
                    if (count == 9999) begin
                        count <= 0;
                    end else begin
                        count <= count + 1;
                    end
                end
            end else begin  // down
                if (tick) begin
                    if (count == 0) begin
                        count <= 9999;
                    end else begin
                        count <= count - 1;
                    end
                end
            end
        end
    end
endmodule

////////////////////////////////////////////////////////////////////////////////////

module clk_div_10hz (
    input  logic clk,
    input  logic reset,
    output logic tick_10hz
);

    logic [$clog2(10_000_000)-1:0] div_counter;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            div_counter <= 0;
            tick_10hz   <= 1'b0;
        end else begin
            if (div_counter == 10_000_000 - 1) begin  //(100MHz / 10M) = 10Hz
                div_counter <= 0;
                tick_10hz   <= 1'b1;
            end else begin
                div_counter <= div_counter + 1;
                tick_10hz   <= 1'b0;
            end
        end
    end
endmodule
```

## fnd_controller
```verilog
`timescale 1ns / 1ps

module fnd_Controller (
    input  logic        clk,
    input  logic        reset,
    input  logic [13:0] number,
    output logic [ 3:0] fndCom,
    output logic [ 7:0] fndFont
);
    logic tick_1khz;
    logic [1:0] count;

    logic [3:0] digit_1, digit_10, digit_100, digit_1000, digit;

    clk_div_1khz U_Clk_Div_1hhz (
        .clk(clk),
        .reset(reset),
        .tich_1khz(tick_1khz)
    );

    counter_2bit U_Counter_2bit (
        .clk  (clk),
        .reset(reset),
        .tick (tick_1khz),
        .count(count)
    );

    decoder_2x4 U_Decoder_2x4 (
        .x(count),
        .y(fndCom)
    );

    digit_splitter U_DigitSplitter (
        .number(number),
        .digit_1(digit_1),
        .digit_10(digit_10),
        .digit_100(digit_100),
        .digit_1000(digit_1000)
    );

    mux_4x1 U_Mux_4x1 (
        .sel(count),
        .x0 (digit_1),
        .x1 (digit_10),
        .x2 (digit_100),
        .x3 (digit_1000),
        .y  (digit)
    );

    BCDtoFND_decoder U_BCDtoFND (
        .bcd(digit),
        .fnd(fndFont)
    );
endmodule

////////////////////////////////////////////////////////////////////////////////////

module clk_div_1khz (
    input  logic clk,
    input  logic reset,
    output logic tich_1khz
);
    logic [$clog2(100_000)-1:0] div_counter;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            div_counter <= 0;
            tich_1khz   <= 0;
        end else begin
            if (div_counter == 100_000 - 1) begin
                div_counter <= 0;
                tich_1khz   <= 1'b1;
            end else begin
                div_counter <= div_counter + 1;
                tich_1khz   <= 0;
            end
        end
    end
endmodule

////////////////////////////////////////////////////////////////////////////////////

module counter_2bit (
    input  logic       clk,
    input  logic       reset,
    input  logic       tick,
    output logic [1:0] count
);
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            count <= 0;
        end else begin
            if (tick) begin
                count <= count + 1;
            end
        end
    end
endmodule

////////////////////////////////////////////////////////////////////////////////////

module decoder_2x4 (
    input  logic [1:0] x,
    output logic [3:0] y
);
    always_comb begin
        y = 4'b1111;
        case (x)
            2'b00: y = 4'b1110;
            2'b01: y = 4'b1101;
            2'b10: y = 4'b1011;
            2'b11: y = 4'b0111;
        endcase
    end
endmodule

////////////////////////////////////////////////////////////////////////////////////

module digit_splitter (
    input  logic [13:0] number,
    output logic [ 3:0] digit_1,
    output logic [ 3:0] digit_10,
    output logic [ 3:0] digit_100,
    output logic [ 3:0] digit_1000
);
    assign digit_1 = (number % 10);
    assign digit_10 = ((number / 10) % 10);
    assign digit_100 = ((number / 100) % 10);
    assign digit_1000 = ((number / 1000) % 10);
endmodule

////////////////////////////////////////////////////////////////////////////////////

module mux_4x1 (
    input  logic [1:0] sel,
    input  logic [3:0] x0,
    input  logic [3:0] x1,
    input  logic [3:0] x2,
    input  logic [3:0] x3,
    output logic [3:0] y
);
    always_comb begin
        y = 4'b0000;
        case (sel)
            2'b00: y = x0;
            2'b01: y = x1;
            2'b10: y = x2;
            2'b11: y = x3;
        endcase
    end
endmodule

////////////////////////////////////////////////////////////////////////////////////

module BCDtoFND_decoder (
    input  logic [3:0] bcd,
    output logic [7:0] fnd
);
    always_comb begin
        case (bcd)
            4'h0: fnd = 8'hc0;
            4'h1: fnd = 8'hf9;
            4'h2: fnd = 8'ha4;
            4'h3: fnd = 8'hb0;
            4'h4: fnd = 8'h99;
            4'h5: fnd = 8'h92;
            4'h6: fnd = 8'h82;
            4'h7: fnd = 8'hf8;
            4'h8: fnd = 8'h80;
            4'h9: fnd = 8'h90;
            4'ha: fnd = 8'h88;
            4'hb: fnd = 8'h83;
            4'hc: fnd = 8'hc6;
            4'hd: fnd = 8'ha1;
            4'he: fnd = 8'h86;
            4'hf: fnd = 8'h8e;
            default: fnd = 8'hff;
        endcase
    end
endmodule

////////////////////////////////////////////////////////////////////////////////////
```

## Btn debounce
```verilog
`timescale 1ns / 1ps

module btn_debounce #(
    parameter int F_COUNT = 10_000  
)(
    input  logic clk,
    input  logic rst,
    input  logic i_btn,
    output logic o_btn
);

    logic [$clog2(F_COUNT)-1:0] r_counter;
    logic                       r_10khz;

    logic [7:0] q_reg, q_next;
    logic       r_edge_q;
    logic       w_debounce;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            r_counter <= '0;
            r_10khz   <= 1'b0;
        end else if (r_counter == F_COUNT - 1) begin
            r_counter <= '0;
            r_10khz   <= 1'b1;
        end else begin
            r_counter <= r_counter + 1;
            r_10khz   <= 1'b0;
        end
    end

    always_ff @(posedge r_10khz or posedge rst) begin
        if (rst)
            q_reg <= 8'b0;
        else
            q_reg <= q_next;
    end

    always_comb begin
        q_next = {i_btn, q_reg[7:1]};
    end

    assign w_debounce = &q_reg;

    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            r_edge_q <= 1'b0;
        else
            r_edge_q <= w_debounce;
    end
    
    assign o_btn = ~w_debounce & r_edge_q;

endmodule
```