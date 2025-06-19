`timescale 1ns / 1ps

module top_sr04 (
    input clk,
    input rst,
    input btn_start,
    input echo,
    output trig,
    output [7:0] fnd_data,
    output [3:0] fnd_com
);

    wire [9:0] w_dist;
    wire w_btn;

    btn_debounce U_Btn_De (
        .clk  (clk),
        .rst  (rst),
        .i_btn(btn_start),
        .o_btn(w_btn)
    );

    sr04_controller U_SR04_CNTL (
        .clk(clk),
        .rst(rst),
        .start(w_btn),
        .echo(echo),
        .trig(trig),
        .dist(w_dist),
        .dist_done()
    );

    fnd_controller U_FND_CNTL (
        .clk(clk),
        .reset(rst),
        .count_data(w_dist),
        .fnd_data(fnd_data),
        .fnd_com(fnd_com)
    );

endmodule
