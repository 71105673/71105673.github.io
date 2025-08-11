---
title: "System Verolog Day-3"
date: "2025-08-08"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# ComportMaster를 통한 UART 통신 제어

**M, m** -> up / down counter를 전환

**R, r** -> Go

**S, s** -> Stop

**C, c** -> Stop 모드일 때, 누르면 Clear

<video controls src="../../../assets/img/CPU/day_3/자막_0808.mp4" title="Title"></video>

## top_uart_counter.sv
```verilog
`timescale 1ns / 1ps

module top_uart_counter (
    input  logic       clk,
    input  logic       reset,
    input  logic       btn_mode,
    input  logic       btn_run_stop,
    input  logic       btn_clear,
    input  logic       rx,
    output logic       tx,
    output logic [1:0] led_mode,
    output logic [1:0] led_run_stop,
    output logic [3:0] fndCom,
    output logic [7:0] fndFont

);
    logic rx_done;
    logic [7:0] rx_data;

    logic com_run, com_stop, com_clear, com_mode;

    uart U_UART (
        .clk(clk),
        .reset(reset),
        .rx(rx),
        .rx_done(rx_done),
        .rx_data(rx_data),
        .tx(tx)
    );

    command_to_uart U_CMD_TO_UART (
        .clk(clk),
        .reset(reset),
        .rx_data_command(rx_data),
        .rx_done_command(rx_done),
        .com_run(com_run),
        .com_stop(com_stop),
        .com_clear(com_clear),
        .com_mode(com_mode)
    );

    top_Module TOP_COUNTER (
        .clk(clk),
        .reset(reset),
        .btn_mode(btn_mode),
        .btn_run_stop(btn_run_stop),
        .btn_clear(btn_clear),
        .com_run(com_run),
        .com_stop(com_stop),
        .com_clear(com_clear),
        .com_mode(com_mode),
        .led_mode(led_mode),
        .led_run_stop(led_run_stop),
        .fndCom(fndCom),
        .fndFont(fndFont)
    );

endmodule
```

## uart.sv
```verilog
`timescale 1ns / 1ps

module uart (
    input  logic       clk,
    input  logic       reset,
    input  logic       rx,
    output logic       rx_done,
    output logic [7:0] rx_data,
    output logic       tx
);

    logic br_tick;
    logic tx_busy, tx_done;

    logic start_tx;

    // start_tx: rx_done 펄스 + transmitter가 idle일 때만 1 클럭짜리 펄스 생성
    assign start_tx = rx_done & ~tx_busy;

    baudrate_gen U_BRAUD_GEN (
        .clk(clk),
        .reset(reset),
        .br_tick(br_tick)
    );

    receive U_Receive (
        .clk(clk),
        .reset(reset),
        .br_tick(br_tick),
        .rx(rx),
        .rx_done(rx_done),
        .rx_data(rx_data)
    );

    transmitter U_Transmitter (
        .clk(clk),
        .reset(reset),
        .br_tick(br_tick),
        .start(start_tx),
        .tx_data(rx_data),
        .tx_busy(tx_busy),
        .tx_done(tx_done),
        .tx(tx)
    );

endmodule

///////////////////////////////////////////////////////////////////////////////

module baudrate_gen (
    input  logic clk,
    input  logic reset,
    output logic br_tick
);

    logic [$clog2((100_000_000 / 9600) / 16) -1:0] br_counter;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            br_counter <= 0;
            br_tick <= 1'b0;
        end else begin
            if (br_counter == ((100_000_000 / 9600) / 16) - 1) begin
                br_counter <= 0;
                br_tick <= 1'b1;
            end else begin
                br_counter <= br_counter + 1;
                br_tick <= 1'b0;
            end
        end
    end

endmodule

//////////////////////////////////////////////////////////////////////////////

module transmitter (
    input  logic       clk,
    input  logic       reset,
    input  logic       br_tick,
    input  logic       start,
    input  logic [7:0] tx_data,
    output logic       tx_busy,
    output logic       tx_done,
    output logic       tx
);
    typedef enum {
        IDLE,
        START,
        DATA,
        STOP
    } tx_state_e;

    tx_state_e tx_state, tx_next_state;

    logic [7:0] temp_data_reg, temp_data_next;
    logic tx_reg, tx_next;
    logic [3:0] tick_cnt_reg, tick_cnt_next;
    logic [2:0] bit_cnt_reg, bit_cnt_next;
    logic tx_done_reg, tx_done_next;
    logic tx_busy_reg, tx_busy_next;

    assign tx       = tx_reg;
    assign tx_busy = tx_busy_reg;
    assign tx_done  = tx_done_reg;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            tx_state      <= IDLE;
            temp_data_reg <= 0;
            tx_reg        <= 1'b1;
            tick_cnt_reg  <= 0;
            bit_cnt_reg   <= 0;
            tx_done_reg   <= 0;
            tx_busy_reg   <= 0;
        end else begin
            tx_state      <= tx_next_state;
            temp_data_reg <= temp_data_next;
            tx_reg        <= tx_next;
            tick_cnt_reg  <= tick_cnt_next;
            bit_cnt_reg   <= bit_cnt_next;
            tx_done_reg   <= tx_done_next;
            tx_busy_reg   <= tx_busy_next;
        end
    end

    always_comb begin
        tx_next_state  = tx_state;
        temp_data_next = temp_data_reg;
        tx_next        = tx_reg;
        tick_cnt_next  = tick_cnt_reg;
        bit_cnt_next   = bit_cnt_reg;
        tx_done_next   = tx_done_reg;
        tx_busy_next   = tx_busy_reg;
        case (tx_state)
            IDLE: begin
                tx_next = 1'b1;
                tx_done_next = 0;
                tx_busy_next = 0;
                if (start) begin
                    tx_next_state  = START;
                    temp_data_next = tx_data;
                    tick_cnt_next  = 0;
                    bit_cnt_next   = 0;
                    tx_busy_next   = 1;
                end
            end
            START: begin
                tx_next = 1'b0;
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tx_next_state = DATA;
                        tick_cnt_next = 0;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            DATA: begin
                tx_next = temp_data_reg[0];
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tick_cnt_next = 0;
                        if (bit_cnt_reg == 7) begin
                            tx_next_state = STOP;
                            bit_cnt_next  = 0;
                        end else begin
                            temp_data_next = {1'b0, temp_data_reg[7:1]};
                            bit_cnt_next   = bit_cnt_reg + 1;
                        end
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            STOP: begin
                tx_next = 1'b1;
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tx_next_state = IDLE;
                        tx_done_next  = 1'b1;
                        tx_busy_next  = 1'b0;
                        tick_cnt_next = 0;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
        endcase
    end
endmodule

///////////////////////////////////////////////////////////////////////////////

module receive (
    input  logic       clk,
    input  logic       reset,
    input  logic       br_tick,  // baud_tick_x16
    input  logic       rx,
    output logic       rx_done,
    output logic [7:0] rx_data
);

    typedef enum {
        IDLE,
        START,
        DATA,
        STOP
    } rx_state_e;

    rx_state_e rx_state, rx_next_state;

    reg rx_done_reg, rx_done_next;
    reg [2:0] bit_cnt_reg, bit_cnt_next;
    reg [4:0] tick_cnt_reg, tick_cnt_next;
    reg [7:0] rx_data_reg, rx_data_next;

    assign rx_done = rx_done_reg;
    assign rx_data = rx_data_reg;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            rx_state     <= IDLE;
            rx_done_reg  <= 0;
            bit_cnt_reg  <= 0;
            tick_cnt_reg <= 0;
            rx_data_reg  <= 0;
        end else begin
            rx_state     <= rx_next_state;
            rx_done_reg  <= rx_done_next;
            bit_cnt_reg  <= bit_cnt_next;
            tick_cnt_reg <= tick_cnt_next;
            rx_data_reg  <= rx_data_next;
        end
    end

    always_comb begin
        rx_next_state = rx_state;
        rx_done_next  = rx_done_reg;
        bit_cnt_next  = bit_cnt_reg;
        tick_cnt_next = tick_cnt_reg;
        rx_data_next  = rx_data_reg;
        case (rx_state)
            IDLE: begin
                tick_cnt_next = 0;
                bit_cnt_next  = 0;
                rx_done_next  = 0;
                if (rx == 1'b0) begin
                    rx_next_state = START;
                end
            end
            START: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 7) begin
                        rx_next_state = DATA;
                        tick_cnt_next = 0;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            DATA: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        rx_data_next[bit_cnt_reg] = rx;
                        tick_cnt_next = 0;
                        if (bit_cnt_reg == 7) begin
                            rx_next_state = STOP;
                            bit_cnt_next  = 0;
                        end else begin
                            rx_next_state = DATA;
                            bit_cnt_next  = bit_cnt_reg + 1;
                        end
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            STOP: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 23) begin
                        rx_done_next  = 1'b1;
                        rx_next_state = IDLE;
                        tick_cnt_next = 0;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
        endcase
    end

endmodule
```

## command_to_uart.sv
```verilog
`timescale 1ns / 1ps

module command_to_uart (
    input  logic       clk,
    input  logic       reset,
    input  logic [7:0] rx_data_command,
    input  logic       rx_done_command,
    output logic       com_run,
    output logic       com_stop,
    output logic       com_clear,
    output logic       com_mode
);
    typedef enum {
        IDLE,
        PROCESS,
        RUN,
        STOP,
        CLEAR,
        MODE
    } command_state_e;

    command_state_e command_state, command_next_state;

    assign com_run   = (command_state == RUN) ? 1 : 0;
    assign com_stop  = (command_state == STOP) ? 1 : 0;
    assign com_clear = (command_state == CLEAR) ? 1 : 0;
    assign com_mode  = (command_state == MODE) ? 1 : 0;


    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            command_state <= IDLE;
        end else begin
            command_state <= command_next_state;
        end
    end

    always_comb begin
        command_next_state = command_state;
        case (command_state)
            IDLE: begin
                if (rx_done_command) begin
                    command_next_state = PROCESS;
                end
            end
            PROCESS: begin
                case (rx_data_command)
                    8'h52, 8'h72: command_next_state = RUN;  //'R', 'r'
                    8'h43, 8'h63: command_next_state = CLEAR;  // 'C', 'c'
                    8'h53, 8'h73: command_next_state = STOP;  // 'S', 'sn_
                    8'h4D, 8'h6D: command_next_state = MODE;  // 'M' or 'm'
                    default: command_next_state = IDLE;
                endcase
            end
            RUN: begin
                command_next_state = IDLE;
            end
            STOP: begin
                command_next_state = IDLE;
            end
            CLEAR: begin
                command_next_state = IDLE;
            end
            MODE: begin
                command_next_state = IDLE;
            end
        endcase
    end
endmodule
```

## top_module.sv (버튼 + counter + fnd 연결)
```verilog
`timescale 1ns / 1ps

module top_Module (
    input  logic       clk,
    input  logic       reset,
    input  logic       btn_mode,
    input  logic       btn_run_stop,
    input  logic       btn_clear,

    input logic       com_run,
    input logic       com_stop,
    input logic       com_clear,
    input logic       com_mode,

    output logic [1:0] led_mode,
    output logic [1:0] led_run_stop,
    output logic [3:0] fndCom,
    output logic [7:0] fndFont
);
    logic [13:0] count;
    logic btn_mode_edge, btn_run_stop_edge, btn_clear_edge;

    button_detector U_BTN_Detector_U (
        .clk(clk),
        .reset(reset),
        .in_button(btn_mode),
        .rising_edge(),
        .falling_dege(btn_mode_edge),
        .both_edge()
    );

    button_detector U_BTN_Detector_R (
        .clk(clk),
        .reset(reset),
        .in_button(btn_run_stop),
        .rising_edge(btn_run_stop_edge),
        .falling_dege(),
        .both_edge()
    );

    button_detector U_BTN_Detector_L (
        .clk(clk),
        .reset(reset),
        .in_button(btn_clear),
        .rising_edge(),
        .falling_dege(btn_clear_edge),
        .both_edge()
    );

    UpDownCounter U_UD_COUNNTER (
        .clk(clk),
        .reset(reset),
        .btn_mode(btn_mode_edge),
        .btn_runstop(btn_run_stop_edge),
        .btn_clear(btn_clear_edge),
        .com_run(com_run),
        .com_stop(com_stop),
        .com_clear(com_clear),
        .com_mode(com_mode),
        .led_mode(led_mode),
        .led_run_stop(led_run_stop),
        .count(count)
    );

    fnd_Controller U_FND_CNT (
        .clk(clk),
        .reset(reset),
        .number(count),
        .fndCom(fndCom),
        .fndFont(fndFont)
    );

endmodule
```

## UpDownCounter.sv
```verilog
`timescale 1ns / 1ps

module UpDownCounter (
    input  logic        clk,
    input  logic        reset,
    input  logic        btn_mode,
    input  logic        btn_runstop,
    input  logic        btn_clear,

    input logic       com_run,
    input logic       com_stop,
    input logic       com_clear,
    input logic       com_mode,

    output logic [ 1:0] led_mode,
    output logic [ 1:0] led_run_stop,
    output logic [13:0] count
);

    logic tick_10hz;
    logic mode_w, run_stop_w, clear_w;

    clk_div_10hz U_Clk_Dib_10hz (
        .clk(clk),
        .reset(reset),
        .run_stop(run_stop_w),
        .clear(clear_w),
        .tick_10hz(tick_10hz)
    );

    up_down_counter U_Up_Down_counter (
        .clk  (clk),
        .reset(reset),
        .tick (tick_10hz),
        .mode (mode_w),
        .clear(clear_w),
        .count(count)
    );

    control_unit U_ControlUnit (
        .clk(clk),
        .reset(reset),
        .btn_mode(btn_mode),
        .btn_run_stop(btn_runstop),
        .btn_clear(btn_clear),

        .com_run(com_run),
        .com_stop(com_stop),
        .com_clear(com_clear),
        .com_mode(com_mode),

        .mode(mode_w),
        .run_stop(run_stop_w),
        .clear(clear_w),
        .led_mode(led_mode),
        .led_run_stop(led_run_stop)
    );

endmodule

////////////////////////////////////////////////////////////////////////////////////

module up_down_counter (
    input  logic        clk,
    input  logic        reset,
    input  logic        tick,
    input  logic        mode,
    input  logic        clear,
    output logic [13:0] count
);
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            count <= 0;
        end else begin
            if (clear) begin
                count <= 0;
            end
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
    input  logic run_stop,
    input  logic clear,
    output logic tick_10hz
);

    logic [$clog2(10_000_000)-1:0] div_counter;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            div_counter <= 0;
            tick_10hz   <= 1'b0;
        end else begin
            if (run_stop) begin
                if (div_counter == 10_000_000 - 1) begin  //(100MHz / 10M) = 10Hz
                    div_counter <= 0;
                    tick_10hz   <= 1'b1;
                end else begin
                    div_counter <= div_counter + 1;
                    tick_10hz   <= 1'b0;
                end
            end
            if (clear) begin
                div_counter <= 0;
                tick_10hz   <= 1'b0;
            end
        end
    end
endmodule

////////////////////////////////////////////////////////////////////////////////////

module control_unit (
    input  logic       clk,
    input  logic       reset,
    input  logic       btn_mode,
    input  logic       btn_run_stop,
    input  logic       btn_clear,
    
    input logic       com_run,
    input logic       com_stop,
    input logic       com_clear,
    input logic       com_mode,

    output logic       mode,
    output logic       run_stop,
    output logic       clear,
    output logic [1:0] led_mode,
    output logic [1:0] led_run_stop
);
    /******************** Mode FSM ************************/
    typedef enum {
        UP,
        DOWN
    } state_mode_e;

    state_mode_e state_mode, next_state_mode;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            state_mode <= UP;
        end else begin
            state_mode <= next_state_mode;
        end
    end

    always_comb begin
        next_state_mode <= state_mode;
        mode = 0;
        led_mode = 2'b00;
        case (state_mode)
            UP: begin
                mode = 0;
                led_mode = 2'b01;
                if (btn_mode  | com_mode) begin
                    next_state_mode <= DOWN;
                end
            end
            DOWN: begin
                mode = 1;
                led_mode = 2'b10;
                if (btn_mode  | com_mode) begin
                    next_state_mode <= UP;
                end
            end
        endcase
    end
    /***************** RUN STOP CLEAR FSM *********************/
    typedef enum {
        STOP,
        RUN,
        CLEAR
    } state_counter_e;

    state_counter_e state_counter, next_state_counter;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            state_counter <= STOP;
        end else begin
            state_counter <= next_state_counter;
        end
    end

    always_comb begin
        next_state_counter <= state_counter;
        run_stop = 0;
        clear = 0;
        led_run_stop = 2'b00;
        case (state_counter)
            STOP: begin
                led_run_stop = 2'b01;
                if (btn_run_stop | com_run) begin
                    next_state_counter <= RUN;
                end else if (btn_clear | com_clear) begin
                    next_state_counter <= CLEAR;
                end
            end
            RUN: begin
                run_stop = 1;
                led_run_stop = 2'b10;
                if (btn_run_stop | com_stop) begin
                    next_state_counter <= STOP;
                end
            end
            CLEAR: begin
                clear = 1;
                led_run_stop = 2'b00;
                next_state_counter <= STOP;
            end
        endcase
    end
endmodule
```




## Uart.sv 모든 신호 version
```verilog
`timescale 1ns / 1ps

module uart (
    // global signal
    input  logic       clk,
    input  logic       reset,
    // transmitter signals
    input  logic       start,
    input  logic [7:0] tx_data,
    output logic       tx_busy,
    output logic       tx_done,
    output logic       tx,
    // receiver signals
    output logic [7:0] rx_data,
    output logic       rx_done,
    input  logic       rx
);

    logic br_tick;

    baudrate_gen U_BRAUD_GEN (
        .clk    (clk),
        .reset  (reset),
        .br_tick(br_tick)
    );

    transmitter U_Transmitter (
        .clk    (clk),
        .reset  (reset),
        .br_tick(br_tick),
        .start  (start),
        .tx_data(tx_data),
        .tx_busy(tx_busy),
        .tx_done(tx_done),
        .tx     (tx)
    );

    receiver U_Receiver (
        .clk(clk),
        .reset(reset),
        .br_tick(br_tick),
        .rx_data(rx_data),
        .rx_done(rx_done),
        .rx(rx)
    );

endmodule

///////////////////////////////////// Baudrate_gen /////////////////////////////////////////

module baudrate_gen (
    input  logic clk,
    input  logic reset,
    output logic br_tick
);
    //logic [$clog2(100_000_000 / 9600 / 16)-1:0] br_counter;
    logic [3:0] br_counter;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            br_counter <= 0;
            br_tick <= 1'b0;
        end else begin
            //if (br_counter == 100_000_000 / 9600 / 16 - 1) begin
            if (br_counter == 10 - 1) begin
                br_counter <= 0;
                br_tick <= 1'b1;
            end else begin
                br_counter <= br_counter + 1;
                br_tick <= 1'b0;
            end
        end
    end

endmodule

///////////////////////////////////// Transmitter /////////////////////////////////////////

module transmitter (
    input  logic       clk,
    input  logic       reset,
    input  logic       br_tick,
    input  logic       start,
    input  logic [7:0] tx_data,
    output logic       tx_busy,
    output logic       tx_done,
    output logic       tx
);
    typedef enum {
        IDLE,
        START,
        DATA,
        STOP
    } tx_state_e;

    tx_state_e tx_state, tx_next_state;
    logic [7:0] temp_data_reg, temp_data_next;
    logic tx_reg, tx_next;
    logic [3:0] tick_cnt_reg, tick_cnt_next;
    logic [2:0] bit_cnt_reg, bit_cnt_next;
    logic tx_done_reg, tx_done_next;
    logic tx_busy_reg, tx_busy_next;

    assign tx = tx_reg;
    assign tx_busy = tx_busy_reg;
    assign tx_done = tx_done_reg;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            tx_state <= IDLE;
            temp_data_reg <= 0;
            tx_reg <= 1'b1;
            tick_cnt_reg <= 0;
            bit_cnt_reg <= 0;
            tx_done_reg <= 0;
            tx_busy_reg <= 0;
        end else begin
            tx_state <= tx_next_state;
            temp_data_reg <= temp_data_next;
            tx_reg <= tx_next;
            tick_cnt_reg <= tick_cnt_next;
            bit_cnt_reg <= bit_cnt_next;
            tx_done_reg <= tx_done_next;
            tx_busy_reg <= tx_busy_next;
        end
    end

    always_comb begin
        tx_next_state = tx_state;
        temp_data_next = temp_data_reg;
        tx_next = tx_reg;
        tick_cnt_next = tick_cnt_reg;
        bit_cnt_next = bit_cnt_reg;
        tx_done_next = tx_done_reg;
        tx_busy_next = tx_busy_reg;
        case (tx_state)
            IDLE: begin
                tx_next = 1'b1;
                tx_done_next = 0;
                tx_busy_next = 0;
                if (start) begin
                    tx_next_state  = START;
                    temp_data_next = tx_data;
                    tick_cnt_next  = 0;
                    bit_cnt_next   = 0;
                    tx_busy_next   = 1;
                end
            end
            START: begin
                tx_next = 1'b0;
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tx_next_state = DATA;
                        tick_cnt_next = 0;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            DATA: begin
                tx_next = temp_data_reg[0];
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tick_cnt_next = 0;
                        if (bit_cnt_reg == 7) begin
                            tx_next_state = STOP;
                            bit_cnt_next  = 0;
                        end else begin
                            temp_data_next = {1'b0, temp_data_reg[7:1]};
                            bit_cnt_next   = bit_cnt_reg + 1;
                        end
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            STOP: begin
                tx_next = 1'b1;
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tx_next_state = IDLE;
                        tx_done_next  = 1;
                        tx_busy_next  = 0;
                        tick_cnt_next = 0;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
        endcase
    end
endmodule

///////////////////////////////////// Reciver /////////////////////////////////////////

module receiver (
    input logic clk,
    input logic reset,
    input logic br_tick,
    output logic [7:0] rx_data,
    output logic rx_done,
    input logic rx
);

    typedef enum {
        IDLE,
        START,
        DATA,
        STOP
    } rx_state_e;

    rx_state_e rx_state, rx_next_state;
    logic [4:0] tick_cnt_reg, tick_cnt_next;
    logic [2:0] bit_cnt_reg, bit_cnt_next;
    logic [7:0] rx_data_reg, rx_data_next;
    logic rx_done_reg, rx_done_next;

    assign rx_data = rx_data_reg;
    assign rx_done = rx_done_reg;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            rx_state <= IDLE;
            tick_cnt_reg <= 0;
            bit_cnt_reg <= 0;
            rx_data_reg <= 0;
            rx_done_reg <= 0;
        end else begin
            rx_state <= rx_next_state;
            tick_cnt_reg <= tick_cnt_next;
            bit_cnt_reg <= bit_cnt_next;
            rx_data_reg <= rx_data_next;
            rx_done_reg <= rx_done_next;
        end
    end

    always_comb begin
        rx_next_state = rx_state;
        rx_done_next  = rx_done;
        tick_cnt_next = tick_cnt_reg;
        bit_cnt_next  = bit_cnt_reg;
        rx_data_next  = rx_data_reg;
        case (rx_state)
            IDLE: begin
                rx_done_next = 0;
                if (rx == 1'b0) begin
                    rx_next_state = START;
                    tick_cnt_next = 0;
                    bit_cnt_next  = 0;
                    rx_data_next  = 0;
                end
            end
            START: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 7) begin // 데이터 가운데서 샘플링
                        tick_cnt_next = 0;
                        rx_next_state = DATA;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            DATA: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tick_cnt_next = 0;
                        rx_data_next  = {rx, rx_data_reg[7:1]};
                        if (bit_cnt_reg == 7) begin
                            bit_cnt_next  = 0;
                            rx_next_state = STOP;
                        end else begin
                            bit_cnt_next = bit_cnt_reg + 1;
                        end
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            STOP: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 23) begin
                        tick_cnt_next = 0;
                        rx_done_next  = 1;
                        rx_next_state = IDLE;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
        endcase
    end
endmodule
```



# Professor

## top_UpDownCounter.sv
```verilog
`timescale 1ns / 1ps

module top_UpDownCounter (
    input  logic       clk,
    input  logic       reset,
    input  logic       btn_mode,
    input  logic       btn_run_stop,
    input  logic       btn_clear,
    output logic [1:0] led_mode,
    output logic [1:0] led_run_stop,
    output logic [3:0] fndCom,
    output logic [7:0] fndFont,
    output logic       tx,
    input  logic       rx
);

    logic [13:0] count;
    logic btn_mode_edge, btn_run_stop_edge, btn_clear_edge;
    logic [7:0] rx_data;
    logic rx_done;

    button_detector U_BTN_MODE (
        .clk         (clk),
        .reset       (reset),
        .i_btn       (btn_mode),
        .rising_edge (),
        .falling_edge(btn_mode_edge),
        .both_edge   ()
    );

    button_detector U_BTN_RUN_STOP (
        .clk         (clk),
        .reset       (reset),
        .i_btn       (btn_run_stop),
        .rising_edge (btn_run_stop_edge),
        .falling_edge(),
        .both_edge   ()
    );

    button_detector U_BTN_CLEAR (
        .clk         (clk),
        .reset       (reset),
        .i_btn       (btn_clear),
        .rising_edge (),
        .falling_edge(btn_clear_edge),
        .both_edge   ()
    );

    uart U_UART (
        .clk    (clk),
        .reset  (reset),
        .start  (rx_done),
        .tx_data(rx_data),
        .tx_busy(),
        .tx_done(),
        .tx     (tx),
        .rx_data(rx_data),
        .rx_done(rx_done),
        .rx     (rx)
    );

    UpDownCounter U_UpDownCounter (
        .clk         (clk),
        .reset       (reset),
        .btn_mode    (btn_mode_edge),
        .btn_run_stop(btn_run_stop_edge),
        .btn_clear   (btn_clear_edge),
        .rx_data     (rx_data),
        .rx_done     (rx_done),
        .led_mode    (led_mode),
        .led_run_stop(led_run_stop),
        .count       (count)
    );

    fndController U_FND_CTRL (
        .clk    (clk),
        .reset  (reset),
        .number (count),
        .fndCom (fndCom),
        .fndFont(fndFont)
    );

endmodule
```

## uart.sv
```verilog
`timescale 1ns / 1ps

module uart (
    // global signal
    input  logic       clk,
    input  logic       reset,
    // transmitter signal
    input  logic       start,
    input  logic [7:0] tx_data,
    output logic       tx_busy,
    output logic       tx_done,
    output logic       tx,
    // receiver signal
    output logic [7:0] rx_data,
    output logic       rx_done,
    input  logic       rx
);

    logic br_tick;

    baudrate_gen U_BRAUD_GEN (
        .clk    (clk),
        .reset  (reset),
        .br_tick(br_tick)
    );

    transmitter U_Transmitter (
        .clk    (clk),
        .reset  (reset),
        .br_tick(br_tick),
        .start  (start),
        .tx_data(tx_data),
        .tx_busy(tx_busy),
        .tx_done(tx_done),
        .tx     (tx)
    );

    receiver U_Receiver (
        .clk    (clk),
        .reset  (reset),
        .br_tick(br_tick),
        .rx_data(rx_data),
        .rx_done(rx_done),
        .rx     (rx)
    );

endmodule


module baudrate_gen (
    input  logic clk,
    input  logic reset,
    output logic br_tick
);

    logic [$clog2(100_000_000 / 9600 / 16)-1:0] br_counter;
    //logic [3:0] br_counter;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            br_counter <= 0;
            br_tick <= 1'b0;
        end else begin
            if (br_counter == 100_000_000 / 9600 / 16 - 1) begin
            //if (br_counter == 10 - 1) begin
                br_counter <= 0;
                br_tick <= 1'b1;
            end else begin
                br_counter <= br_counter + 1;
                br_tick <= 1'b0;
            end
        end
    end

endmodule


module transmitter (
    input  logic       clk,
    input  logic       reset,
    input  logic       br_tick,
    input  logic       start,
    input  logic [7:0] tx_data,
    output logic       tx_busy,
    output logic       tx_done,
    output logic       tx
);

    typedef enum {
        IDLE,
        START,
        DATA,
        STOP
    } tx_state_e;

    tx_state_e tx_state, tx_next_state;
    logic [7:0] temp_data_reg, temp_data_next;
    logic tx_reg, tx_next;
    logic [3:0] tick_cnt_reg, tick_cnt_next;
    logic [2:0] bit_cnt_reg, bit_cnt_next;
    logic tx_done_reg, tx_done_next;
    logic tx_busy_reg, tx_busy_next;

    assign tx = tx_reg;
    assign tx_busy = tx_busy_reg;
    assign tx_done = tx_done_reg;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            tx_state <= IDLE;
            temp_data_reg <= 0;
            tx_reg <= 1'b1;
            tick_cnt_reg <= 0;
            bit_cnt_reg <= 0;
            tx_done_reg <= 0;
            tx_busy_reg <= 0;
        end else begin
            tx_state <= tx_next_state;
            temp_data_reg <= temp_data_next;
            tx_reg <= tx_next;
            tick_cnt_reg <= tick_cnt_next;
            bit_cnt_reg <= bit_cnt_next;
            tx_done_reg <= tx_done_next;
            tx_busy_reg <= tx_busy_next;
        end
    end

    always_comb begin
        tx_next_state = tx_state;
        temp_data_next = temp_data_reg;
        tx_next = tx_reg;
        tick_cnt_next = tick_cnt_reg;
        bit_cnt_next = bit_cnt_reg;
        tx_done_next = tx_done_reg;
        tx_busy_next = tx_busy_reg;
        case (tx_state)
            IDLE: begin
                tx_next = 1'b1;
                tx_done_next = 0;
                tx_busy_next = 0;
                if (start) begin
                    tx_next_state  = START;
                    temp_data_next = tx_data;
                    tick_cnt_next  = 0;
                    bit_cnt_next   = 0;
                    tx_busy_next   = 1;
                end
            end
            START: begin
                tx_next = 1'b0;
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tx_next_state = DATA;
                        tick_cnt_next = 0;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            DATA: begin
                tx_next = temp_data_reg[0];
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tick_cnt_next = 0;
                        if (bit_cnt_reg == 7) begin
                            tx_next_state = STOP;
                            bit_cnt_next  = 0;
                        end else begin
                            temp_data_next = {1'b0, temp_data_reg[7:1]};
                            bit_cnt_next   = bit_cnt_reg + 1;
                        end
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            STOP: begin
                tx_next = 1'b1;
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tx_next_state = IDLE;
                        tx_done_next  = 1;
                        tx_busy_next  = 0;
                        tick_cnt_next = 0;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
        endcase
    end

endmodule


module receiver (
    input  logic       clk,
    input  logic       reset,
    input  logic       br_tick,
    output logic [7:0] rx_data,
    output logic       rx_done,
    input  logic       rx
);

    typedef enum {
        IDLE,
        START,
        DATA,
        STOP
    } rx_state_e;

    rx_state_e rx_state, rx_next_state;

    logic [4:0] tick_cnt_reg, tick_cnt_next;
    logic [2:0] bit_cnt_reg, bit_cnt_next;
    logic [7:0] rx_data_reg, rx_data_next;
    logic rx_done_next, rx_done_reg;

    assign rx_data = rx_data_reg;
    assign rx_done = rx_done_reg;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            rx_state     <= IDLE;
            tick_cnt_reg <= 0;
            bit_cnt_reg  <= 0;
            rx_data_reg  <= 0;
            rx_done_reg  <= 0;

        end else begin
            rx_state     <= rx_next_state;
            tick_cnt_reg <= tick_cnt_next;
            bit_cnt_reg  <= bit_cnt_next;
            rx_data_reg  <= rx_data_next;
            rx_done_reg  <= rx_done_next;
        end
    end

    always_comb begin
        rx_next_state = rx_state;
        rx_done_next  = rx_done;
        tick_cnt_next = tick_cnt_reg;
        bit_cnt_next  = bit_cnt_reg;
        rx_data_next  = rx_data_reg;
        case (rx_state)
            IDLE: begin
                rx_done_next = 0;
                if (rx == 1'b0) begin
                    rx_next_state = START;
                    tick_cnt_next = 0;
                    bit_cnt_next  = 0;
                    rx_data_next  = 0;
                end
            end
            START: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 7) begin
                        tick_cnt_next = 0;
                        rx_next_state = DATA;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            DATA: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tick_cnt_next = 0;
                        rx_data_next  = {rx, rx_data_reg[7:1]};
                        if (bit_cnt_reg == 7) begin
                            bit_cnt_next  = 0;
                            rx_next_state = STOP;
                        end else begin
                            bit_cnt_next = bit_cnt_reg + 1;
                        end
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            STOP: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 23) begin
                        tick_cnt_next = 0;
                        rx_done_next  = 1;
                        rx_next_state = IDLE;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
        endcase
    end

endmodule
```

## UpDownCounter.sv
```verilog
`timescale 1ns / 1ps

module UpDownCounter (
    input  logic        clk,
    input  logic        reset,
    input  logic        btn_mode,
    input  logic        btn_run_stop,
    input  logic        btn_clear,
    input  logic [ 7:0] rx_data,
    input  logic        rx_done,
    output logic [ 1:0] led_mode,
    output logic [ 1:0] led_run_stop,
    output logic [13:0] count
);

    logic tick_10hz;
    logic mode, run_stop, clear;

    clk_div_10hz U_CLK_DIV_10hz (
        .clk      (clk),
        .reset    (reset),
        .run_stop (run_stop),
        .clear    (clear),
        .tick_10hz(tick_10hz)
    );

    up_down_counter U_UP_DOWN_COUNTER (
        .clk  (clk),
        .reset(reset),
        .tick (tick_10hz),
        .mode (mode),
        .clear(clear),
        .count(count)
    );

    control_unit U_CU (
        .clk         (clk),
        .reset       (reset),
        .btn_mode    (btn_mode),
        .btn_run_stop(btn_run_stop),
        .btn_clear   (btn_clear),
        .rx_data     (rx_data),
        .rx_done     (rx_done),
        .mode        (mode),
        .run_stop    (run_stop),
        .clear       (clear),
        .led_mode    (led_mode),
        .led_run_stop(led_run_stop)
    );

endmodule


module up_down_counter (
    input  logic        clk,
    input  logic        reset,
    input  logic        tick,
    input  logic        mode,
    input  logic        clear,
    output logic [13:0] count
);

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            count <= 0;
        end else begin
            if (clear) begin
                count <= 0;
            end
            if (mode == 1'b0) begin  // up counter
                if (tick) begin
                    if (count == 9999) begin
                        count <= 0;
                    end else begin
                        count <= count + 1;
                    end
                end
            end else begin  // down counter
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


module clk_div_10hz (
    input  logic clk,
    input  logic reset,
    input  logic run_stop,
    input  logic clear,
    output logic tick_10hz
);

    //logic [23:0] div_counter;
    logic [$clog2(10_000_000)-1:0] div_counter;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            div_counter <= 0;
            tick_10hz   <= 1'b0;
        end else begin
            if (run_stop) begin
                if (div_counter == 10_000_000 - 1) begin
                    div_counter <= 0;
                    tick_10hz   <= 1'b1;
                end else begin
                    div_counter <= div_counter + 1;
                    tick_10hz   <= 1'b0;
                end
            end
            if (clear) begin
                div_counter <= 0;
                tick_10hz   <= 1'b0;
            end
        end
    end

endmodule


module control_unit (
    input  logic       clk,
    input  logic       reset,
    input  logic       btn_mode,
    input  logic       btn_run_stop,
    input  logic       btn_clear,
    input  logic [7:0] rx_data,
    input  logic       rx_done,
    output logic       mode,
    output logic       run_stop,
    output logic       clear,
    output logic [1:0] led_mode,
    output logic [1:0] led_run_stop
);

    /************************** MODE FSM **************************/

    typedef enum {
        UP,
        DOWN
    } state_mode_e;

    state_mode_e state_mode, next_state_mode;

    // transition logic
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            state_mode <= UP;
        end else begin
            state_mode <= next_state_mode;
        end
    end

    // output logic
    always_comb begin
        next_state_mode = state_mode;
        mode = 0;
        led_mode = 2'b00;
        case (state_mode)
            UP: begin
                led_mode = 2'b01;
                mode = 0;
                if (btn_mode) begin
                    next_state_mode = DOWN;
                end
                if (rx_done) begin
                    if (rx_data == 8'h4d || rx_data == 8'h6d) begin  // M, m
                        next_state_mode = DOWN;
                    end
                end
            end
            DOWN: begin
                led_mode = 2'b10;
                mode = 1;
                if (btn_mode) begin
                    next_state_mode = UP;
                end
                if (rx_done) begin
                    if (rx_data == 8'h4d || rx_data == 8'h6d) begin  // M, m
                        next_state_mode = UP;
                    end
                end
            end
        endcase
    end

    /********************* RUN STOP CLEAR FSM *********************/

    typedef enum {
        STOP,
        RUN,
        CLEAR
    } state_counter_e;

    state_counter_e state_counter, next_state_counter;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            state_counter <= STOP;
        end else begin
            state_counter <= next_state_counter;
        end
    end

    always_comb begin
        next_state_counter = state_counter;
        run_stop = 0;
        clear = 0;
        led_run_stop = 2'b00;
        case (state_counter)
            STOP: begin
                led_run_stop = 2'b01;
                if (btn_run_stop) next_state_counter = RUN;
                else if (btn_clear) next_state_counter = CLEAR;
                if (rx_done) begin
                    if (rx_data == 8'h52 || rx_data == 8'h72) begin  // R, r
                        next_state_counter = RUN;
                    end
                end
                if (rx_done) begin
                    if (rx_data == 8'h43 || rx_data == 8'h63) begin  // C, c
                        next_state_counter = CLEAR;
                    end
                end
            end
            RUN: begin
                run_stop = 1;
                led_run_stop = 2'b10;
                if (btn_run_stop) next_state_counter = STOP;
                if (rx_done) begin
                    if (rx_data == 8'h53 || rx_data == 8'h73) begin  // S, s
                        next_state_counter = STOP;
                    end
                end
            end
            CLEAR: begin
                clear = 1;
                next_state_counter = STOP;
            end
        endcase
    end

endmodule
```


## fndController.sv
```verilog
`timescale 1ns / 1ps

module fndController (
    input  logic        clk,
    input  logic        reset,
    input  logic [13:0] number,
    output logic [ 3:0] fndCom,
    output logic [ 7:0] fndFont
);

    logic       tick_1khz;
    logic [1:0] count;
    logic [3:0] digit, digit_1, digit_10, digit_100, digit_1000;

    clk_div_1khz U_CLK_DIV_1khz (
        .clk      (clk),
        .reset    (reset),
        .tick_1khz(tick_1khz)
    );

    counter_2bit U_COUNTER_2BIT (
        .clk  (clk),
        .reset(reset),
        .tick (tick_1khz),
        .count(count)
    );

    decoder_2x4 U_DECODER_2x4 (
        .x(count),
        .y(fndCom)
    );

    digitSplitter U_DIGITSPLITTER (
        .number    (number),
        .digit_1   (digit_1),
        .digit_10  (digit_10),
        .digit_100 (digit_100),
        .digit_1000(digit_1000)
    );

    mux_4x1 U_MUX_4x1 (
        .sel(count),
        .x0 (digit_1),
        .x1 (digit_10),
        .x2 (digit_100),
        .x3 (digit_1000),
        .y  (digit)
    );

    BCDtoFND_Decoder U_BCDtoFND (
        .bcd(digit),
        .fnd(fndFont)
    );

endmodule


module clk_div_1khz (
    input  logic clk,
    input  logic reset,
    output logic tick_1khz
);

    logic [$clog2(100_000)-1:0] div_counter;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            div_counter <= 0;
            tick_1khz   <= 1'b0;
        end else begin
            if (div_counter == 100_000 - 1) begin
                div_counter <= 0;
                tick_1khz   <= 1'b1;
            end else begin
                div_counter <= div_counter + 1;
                tick_1khz   <= 1'b0;
            end
        end
    end

endmodule


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


module digitSplitter (
    input  logic [13:0] number,
    output logic [ 3:0] digit_1,
    output logic [ 3:0] digit_10,
    output logic [ 3:0] digit_100,
    output logic [ 3:0] digit_1000
);

    assign digit_1    = number % 10;
    assign digit_10   = number / 10 % 10;
    assign digit_100  = number / 100 % 10;
    assign digit_1000 = number / 1000 % 10;

endmodule


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


module BCDtoFND_Decoder (
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
```


## button_detector.sv
```verilog
`timescale 1ns / 1ps

module button_detector (
    input  logic clk,
    input  logic reset,
    input  logic i_btn,
    output logic rising_edge,
    output logic falling_edge,
    output logic both_edge
);

    logic                       clk_1khz;
    logic                       debounce;
    logic [                7:0] shift_reg;
    logic [$clog2(100_000)-1:0] div_counter;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            div_counter <= 0;
            clk_1khz    <= 1'b0;
        end else begin
            if (div_counter == 100_000 - 1) begin
                div_counter <= 0;
                clk_1khz    <= 1'b1;
            end else begin
                div_counter <= div_counter + 1;
                clk_1khz    <= 1'b0;
            end
        end
    end

    shift_register U_SHIFT_REG (
        .clk   (clk_1khz),
        .reset (reset),
        .i_data(i_btn),
        .o_data(shift_reg)
    );

    assign debounce = &shift_reg;
    //assign o_btn = debounce;

    logic [1:0] edge_reg;

    always @(posedge clk, posedge reset) begin
        if (reset) begin
            edge_reg <= 0;
        end else begin
            edge_reg[0] <= debounce;
            edge_reg[1] <= edge_reg[0];
        end
    end

    assign rising_edge  = edge_reg[0] & ~edge_reg[1];
    assign falling_edge = ~edge_reg[0] & edge_reg[1];
    assign both_edge    = rising_edge | falling_edge;

endmodule


module shift_register (
    input  logic       clk,
    input  logic       reset,
    input  logic       i_data,
    output logic [7:0] o_data
);

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            o_data <= 0;
        end else begin
            o_data <= {i_data, o_data[7:1]};  // right shift
            //o_data <= {o_data[6:0], i_data};// left shift
        end
    end

endmodule
```