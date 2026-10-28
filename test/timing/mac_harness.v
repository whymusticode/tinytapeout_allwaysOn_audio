/*
 * Timing harness for mac.v: every MAC input comes from a flip-flop, as it will in the engine
 * (activation buffer read register, weight shift register), so the paths timed are the real
 * register -> multiply -> add -> accumulator paths. Not part of the chip.
 */

`default_nettype none

module mac_harness #(
    parameter ACC_W = 24,
    parameter PIPE  = 0
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             en,
    input  wire             clr,
    input  wire [      7:0] a,
    input  wire [      7:0] w,
    output wire [ACC_W-1:0] acc
);

  reg en_q, clr_q;
  reg [7:0] a_q, w_q;
  always @(posedge clk) begin
    en_q  <= en;
    clr_q <= clr;
    a_q   <= a;
    w_q   <= w;
  end

  mac #(.ACC_W(ACC_W), .PIPE(PIPE)) u_mac (
      .clk(clk), .rst_n(rst_n), .en(en_q), .clr(clr_q), .a(a_q), .w(w_q), .acc(acc));

endmodule
