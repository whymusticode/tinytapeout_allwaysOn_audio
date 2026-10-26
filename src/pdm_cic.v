/*
 * Copyright (c) 2026 Michael Benton
 * SPDX-License-Identifier: Apache-2.0
 *
 * PDM microphone interface + 3rd-order CIC decimator.
 *
 *   clk (16.384 MHz) / CLK_DIV -> pdm_clk (1.024 MHz)
 *   1-bit PDM at 1.024 MHz / DECIM (64) -> 16 kHz signed 16-bit PCM
 *
 * pdm_edge selects which half of the PDM clock the mic drives data in:
 *   0: data valid while pdm_clk is high (sampled just before the falling edge)
 *   1: data valid while pdm_clk is low  (sampled just before the rising edge)
 * Stereo mics drive one channel per half, so this also picks the channel.
 *
 * Bit-exact reference: model/frontend.py:cic3()
 */

`default_nettype none

module pdm_cic #(
    parameter CLK_DIV_LOG2 = 4,  // pdm_clk = clk / 2^CLK_DIV_LOG2
    parameter DECIM_LOG2   = 6,  // R = 64
    parameter W            = 20  // 1 + N*log2(R) = 19, +1 so +R^N fits
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        pdm_data,
    input  wire        pdm_edge,
    output wire        pdm_clk,
    output reg  [15:0] pcm,
    output reg         pcm_valid
);

  reg [CLK_DIV_LOG2-1:0] div_cnt;
  assign pdm_clk = div_cnt[CLK_DIV_LOG2-1];

  // Two-flop synchronizer for the asynchronous mic data pin.
  reg [1:0] sync;

  wire sample_now = pdm_edge ? (div_cnt == {1'b0, {(CLK_DIV_LOG2 - 1) {1'b1}}})
                             : (div_cnt == {CLK_DIV_LOG2{1'b1}});

  // Integrators run at the PDM rate. Input bit maps to +1 / -1.
  reg signed [W-1:0] i1, i2, i3;
  wire signed [W-1:0] x = sync[1] ? {{(W - 1) {1'b0}}, 1'b1} : {W{1'b1}};

  reg [DECIM_LOG2-1:0] dec_cnt;
  reg                  dec_pending;

  // Combs run at the output rate.
  reg signed [W-1:0] i3_d, c1_d, c2_d;
  wire signed [W-1:0] c1 = i3 - i3_d;
  wire signed [W-1:0] c2 = c1 - c1_d;
  wire signed [W-1:0] c3 = c2 - c2_d;

  // Scale to 16 bits: c3 is in [-2^18, 2^18]; >>> 3 gives [-2^15, 2^15].
  wire signed [W-4:0] c3_s = c3[W-1:3];
  wire sat_hi = (c3_s > 32767);
  wire sat_lo = (c3_s < -32768);

  always @(posedge clk) begin
    if (!rst_n) begin
      div_cnt     <= 0;
      sync        <= 0;
      i1          <= 0;
      i2          <= 0;
      i3          <= 0;
      dec_cnt     <= 0;
      dec_pending <= 0;
      i3_d        <= 0;
      c1_d        <= 0;
      c2_d        <= 0;
      pcm         <= 0;
      pcm_valid   <= 0;
    end else begin
      div_cnt   <= div_cnt + 1'b1;
      sync      <= {sync[0], pdm_data};
      pcm_valid <= 0;

      if (sample_now) begin
        // Pipelined integrators (each uses the previous stage's old value).
        i1      <= i1 + x;
        i2      <= i2 + i1;
        i3      <= i3 + i2;
        dec_cnt <= dec_cnt + 1'b1;
        if (&dec_cnt) dec_pending <= 1;
      end

      // One cycle after the 64th integrator update, i3 holds the new value.
      if (dec_pending) begin
        dec_pending <= 0;
        i3_d        <= i3;
        c1_d        <= c1;
        c2_d        <= c2;
        pcm         <= sat_hi ? 16'h7fff : sat_lo ? 16'h8000 : c3_s[15:0];
        pcm_valid   <= 1;
      end
    end
  end

endmodule
