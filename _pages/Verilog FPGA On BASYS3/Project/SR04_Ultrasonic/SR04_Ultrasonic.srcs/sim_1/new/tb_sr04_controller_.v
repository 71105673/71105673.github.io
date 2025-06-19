`timescale 1ns / 1ps

module tb_sr04_controller_ ();
    reg clk;
    reg rst;
    reg start;
    reg echo;
    wire trig;
    wire [9:0] dist;
    wire dist_don;

    sr04_controller dut(
        .clk(clk),
        .rst(rst),
        .start(start),
        .echo(echo),
        .trig(trig),
        .dist(dist),
        .dist_done(dist_don)
    );

    always #5 clk = ~clk;

     initial begin
        // Initial state
        clk = 0;
        rst = 1;
        start = 0;
        echo = 0;

        // Reset
        #20;
        rst = 0;

        // Trigger start after 100ns
        #100;
        start = 1;
        #10;
        start = 0;

        // Simulate echo HIGH after 500ns, keep HIGH for 1740ns (simulating 30cm)
        // echo HIGH duration = 1740us → 30cm (30 * 58 = 1740us)
        #500;
        echo = 1;
        #(5800 * 20);
        echo = 0;

        // Wait for done
        #1000;

        // Second measurement
        #2000;
        start = 1;
        #10;
        start = 0;

        #700;
        echo = 1;
        #(5800 * 10); // 20cm
        echo = 0;

        #1000;

        $stop;
    end

endmodule
