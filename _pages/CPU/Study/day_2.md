---
title: "System Verolog Day-1"
date: "2025-08-07"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# U, R, L 버튼으로 기능을 구현한 counter

**BtnU** -> up / down counter를 전환

**BtnR** -> Go / Stop 전환

**BtnL** -> Stop 모드일 때, 누르면 Clear

**BtnC** -> 리셋 기능

<video controls src="../../../assets/img/CPU/0807자막버전.mp4" title="Title"></video>

## top_updowncounter.sv
```verilog
`timescale 1ns / 1ps

module top_UpDownCounter(
    input logic clk,
    input logic reset,
    input logic Ubtn,
    input logic Lbtn,
    input logic Rbtn,
    output logic [3:0] fndCom,
    output logic [7:0] fndFont,
    output [3:0] led
);
    logic [13:0] count;
    logic button_edge_U, button_edge_R, button_edge_L;
    logic up_down_mode, o_runstop, o_clear;

    ////////////// 버튼 디바운스 //////////////
    button_detector U_UBTN_Detector(
        .clk(clk),
        .reset(reset),
        .in_button(Ubtn),
        .rising_edge(),
        .falling_dege(button_edge_U),
        .both_edge()
    );

    button_detector R_UBTN_Detector(
        .clk(clk),
        .reset(reset),
        .in_button(Rbtn),
        .rising_edge(),
        .falling_dege(button_edge_R),
        .both_edge()
    );

    button_detector L_UBTN_Detector(
        .clk(clk),
        .reset(reset),
        .in_button(Lbtn),
        .rising_edge(),
        .falling_dege(button_edge_L),
        .both_edge()
    );
    //////////////////////////////////////////

    ///////////////// CU ////////////////////

    control_unit_updown U_UPDOWN_CU(
        .clk(clk),
        .reset(reset),
        .button(button_edge_U),
        .mode(up_down_mode)
    );

    control_unit_runstopclear U_RUNSTOPCLEAR_CU(
        .clk(clk),
        .rst(reset),
        .i_btn_runstop(button_edge_R),
        .i_btn_clear(button_edge_L),
        .o_runstop(o_runstop),
        .o_clear(o_clear)
    );

    /////////////////////////////////////////

    ///////////////// DP ////////////////////

    UpDownCounter U_UD_COUNNTER(
        .clk(clk),
        .reset(reset),
        .mode_change(up_down_mode),
        .run_stop(o_runstop),
        .clear(o_clear),
        .count(count)
    );

    fnd_Controller U_FND_CNT(
        .clk(clk),
        .reset(reset),
        .number(count),
        .fndCom(fndCom),
        .fndFont(fndFont)
    );
    /////////////////////////////////////////
    
    LED U_LED(
        .up_down_mode(up_down_mode),
        .o_runstop(o_runstop),
        .o_clear(o_clear),
        .led_out(led)
    );

endmodule



module LED (
    input logic up_down_mode,
    input logic o_runstop,
    input logic o_clear,
    output logic [3:0] led_out
);
    always_comb begin
        led_out = 4'b0000;

        // up/down 모드 표시
        led_out[0] = (up_down_mode == 1'b0); // UP일 때 ON
        led_out[1] = (up_down_mode == 1'b1); // DOWN일 때 ON

        // run/stop 표시
        led_out[2] = (o_runstop == 1'b0); // STOP일 때 ON
        led_out[3] = (o_runstop == 1'b1); // RUN일 때 ON
    end

endmodule
```

## button_detector.sv
```verilog
`timescale 1ns / 1ps

module button_detector (
    input  logic clk,
    input  logic reset,
    input  logic in_button,
    output logic rising_edge,
    output logic falling_dege,
    output logic both_edge
);

    logic clk_1khz;
    logic [$clog2(100_000)-1:0] div_counter;

    logic debounce;
    logic [7:0] sh_reg;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            div_counter <= 0;
            clk_1khz <= 0;
        end else begin
            if (div_counter == 100_000 - 1) begin
                div_counter <= 0;
                clk_1khz <= 1;
            end else begin
                div_counter <= div_counter + 1;
                clk_1khz <= 1'b0;
            end
        end
    end

    shift_register U_Shift_Register (
        .clk     (clk_1khz),
        .reset   (reset),
        .in_data (in_button),
        .out_data(sh_reg)
    );

    assign debounce = &sh_reg;
    logic [1:0] edge_reg;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            edge_reg <= 0;
        end else begin
            edge_reg[0] <= debounce;
            edge_reg[1] <= edge_reg[0];
        end
    end

    assign rising_edge = edge_reg[0] & ~edge_reg[1];
    assign falling_dege = ~edge_reg[0] & edge_reg[1];
    assign both_edge = rising_edge | falling_dege;  //눌러도 떼도
endmodule

/////////////////////////////////////////////////////////////////////////////////////////

module shift_register (
    input  logic       clk,
    input  logic       reset,
    input  logic       in_data,
    output logic [7:0] out_data
);

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            out_data <= 0;
        end else begin
            out_data <= {in_data, out_data[7:1]};  // right shift
            // out_data <= {out_data[6:0], in_data};   // left shift
        end
    end
endmodule

/////////////////////////////////////////////////////////////////////////////////////////
```

## updowncounter.sv
```verilog
`timescale 1ns / 1ps

module UpDownCounter (
    input  logic        clk,
    input  logic        reset,
    input  logic        mode_change,
    input  logic        run_stop,
    input  logic        clear,
    output logic [13:0] count
);
    logic tick_10hz;

    clk_div_10hz U_Clk_Dib_10hz (
        .clk(clk),
        .reset(reset),
        .tick_10hz(tick_10hz)
    );

    up_down_counter U_Up_Down_counter (
        .clk  (clk),
        .reset(reset | clear),
        .tick (tick_10hz & run_stop),
        .mode (mode_change),
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

## control_unit_updown.sv
```verilog
`timescale 1ns / 1ps

module control_unit_updown (
    input  logic clk,
    input  logic reset,
    input  logic button,
    output logic mode
);
    typedef enum {
        UP,
        DOWN
    } state_e;

    state_e state, next_state;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            state <= UP;
        end else begin
            state <= next_state;
        end
    end

    always_comb begin
        next_state <= state;
        mode = 0;
        case (state)
            UP: begin
                mode = 0;
                if (button) begin
                    next_state <= DOWN;
                end
            end
            DOWN: begin
                mode = 1;
                if (button) begin
                    next_state <= UP;
                end
            end
        endcase
    end
endmodule

module control_unit_runstopclear (
    input  logic clk,
    input  logic rst,
    input  logic i_btn_runstop,
    input  logic i_btn_clear,
    output logic o_runstop,
    output logic o_clear
);
    typedef enum logic [1:0] {
        STOP  = 2'b00,
        RUN   = 2'b01,
        CLEAR = 2'b10
    } state_e;

    state_e state, next;

    assign o_clear   = (state == CLEAR) ? 1 : 0;
    assign o_runstop = (state == RUN) ? 1 : 0;

    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            state <= STOP;
        else
            state <= next;
    end

    always_comb begin
        next = state;
        case (state)
            STOP: begin
                if (i_btn_runstop)
                    next = RUN;
                else if (i_btn_clear)
                    next = CLEAR;
            end
            RUN: begin
                if (i_btn_runstop)
                    next = STOP;
            end
            CLEAR: begin
                if (i_btn_clear == 0)
                    next = STOP;
            end
            default: next = STOP;
        endcase
    end

endmodule
```