/*
 * Copyright (c) 2026 Michael Benton
 * SPDX-License-Identifier: Apache-2.0
 *
 * Engine multiply-accumulate: one signed 8x8 product per clock into a wrapping accumulator.
 *
 *   en & clr:  acc = a * w            (first term of a new dot product)
 *   en & !clr: acc = acc + a * w
 *   !en & clr: acc = 0
 *
 * 4-bit weights are sign-extended to 8 bits by the weight reader, so the MAC is the same for
 * both weight widths. ACC_W = 24 holds any dot product of up to 511 terms exactly
 * (|a * w| <= 2^14); each doubling of the fan-in needs one more bit.
 *
 * PIPE = 0: a, w -> product -> add -> acc in one clock.
 * PIPE = 1: the product is registered first, so acc lags one clock; same throughput.
 * Timing at 16.384 MHz on gf180mcuD 7-track cells: ASIC_VERIFY.md, "MAC timing".
 *
 * Bit-exact reference: test/unit/test_mac.py
 */

`default_nettype none

module mac #(
    parameter AW    = 8,
    parameter WW    = 8,
    parameter ACC_W = 24,
    parameter PIPE  = 0
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    en,
    input  wire                    clr,
    input  wire signed [   AW-1:0] a,
    input  wire signed [   WW-1:0] w,
    output reg  signed [ACC_W-1:0] acc
);

  localparam PW = AW + WW;

  wire signed [PW-1:0] prod = a * w;

  wire signed [PW-1:0] p;
  wire                 p_en, p_clr;

  generate
    if (PIPE) begin : g_pipe
      reg signed [PW-1:0] p_q;
      reg                 en_q, clr_q;
      always @(posedge clk) begin
        if (!rst_n) begin
          p_q   <= 0;
          en_q  <= 1'b0;
          clr_q <= 1'b0;
        end else begin
          p_q   <= prod;
          en_q  <= en;
          clr_q <= clr;
        end
      end
      assign p     = p_q;
      assign p_en  = en_q;
      assign p_clr = clr_q;
    end else begin : g_comb
      assign p     = prod;
      assign p_en  = en;
      assign p_clr = clr;
    end
  endgenerate

  wire signed [ACC_W-1:0] p_ext = {{(ACC_W - PW) {p[PW-1]}}, p};
  wire signed [ACC_W-1:0] base = p_clr ? {ACC_W{1'b0}} : acc;

  always @(posedge clk) begin
    if (!rst_n) acc <= 0;
    else if (p_en) acc <= base + p_ext;
    else if (p_clr) acc <= 0;
  end

endmodule
