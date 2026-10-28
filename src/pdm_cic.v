/*
 * Copyright (c) 2026 Michael Benton
 * SPDX-License-Identifier: Apache-2.0
 *
 * PDM microphone interface + 3rd-order CIC decimator.
 *
 * Two clock sources for the mic:
 *   ext_clk = 0: we drive the mic, pdm_clk = clk / 2^CLK_DIV_LOG2 (1.024 MHz at 16.384 MHz).
 *   ext_clk = 1: something else drives it (a phone's codec, listening in on its mic): the mic
 *                clock comes in on pdm_clk_in. It is synchronised like the data and its edges
 *                are detected in the clk domain, so it must be slower than clk / 5
 *                (3.2 MHz at 16.384 MHz): the data bit is taken from the last sample before the
 *                edge was seen, up to one clk period before the real edge, so the mic's data must
 *                be valid from (half a PDM period - 1 clk) before the sampling edge.
 * decim = R (2..255): PCM rate = PDM rate / R; 16 kHz needs R = 64 at 1.024 MHz, 96 at 1.536,
 *   128 at 2.048, 150 at 2.4, 192 at 3.072 MHz.
 * shift: PCM = sat16(c3 >>> shift). The CIC gain is R^3, so shift = ceil(3 log2 R) - 15 keeps full
 *   scale (3 at R = 64, 6 at 128, 8 at 192); non-power-of-2 R leaves a fixed level offset
 *   (< 6 dB), which the model's level augmentation covers.
 *
 * pdm_edge selects which half of the PDM clock the mic drives data in:
 *   0: data valid while pdm_clk is high (sampled just before the falling edge)
 *   1: data valid while pdm_clk is low  (sampled just before the rising edge)
 * Stereo mics drive one channel per half, so this also picks the channel.
 *
 * The comb runs one stage per clock after each decimation point (c1, c2, c3, then saturate):
 * PCM is needed once per R PDM bits, and a single-clock comb + saturation missed timing at the
 * slow corner (ASIC_VERIFY.md).
 *
 * Bit-exact reference: model/frontend.py:cic3()
 */

`default_nettype none

module pdm_cic #(
    parameter CLK_DIV_LOG2 = 4,  // internal pdm_clk = clk / 2^CLK_DIV_LOG2
    parameter W            = 25  // 1 + 3 * log2(255) -> 25 bits, enough for any R up to 255
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        pdm_data,
    input  wire        pdm_edge,
    input  wire        ext_clk,
    input  wire        pdm_clk_in,
    input  wire [ 7:0] decim,
    input  wire [ 3:0] shift,
    output wire        pdm_clk,
    output reg  [15:0] pcm,
    output reg         pcm_valid
);

  reg [CLK_DIV_LOG2-1:0] div_cnt;
  assign pdm_clk = ext_clk ? 1'b0 : div_cnt[CLK_DIV_LOG2-1];

  // Synchronizers: data and external clock go through identical chains, so sync[2] and
  // ck[2] were sampled at the same instant.
  reg [2:0] sync, ck;

  wire int_sample = pdm_edge ? (div_cnt == {1'b0, {(CLK_DIV_LOG2 - 1) {1'b1}}})
                             : (div_cnt == {CLK_DIV_LOG2{1'b1}});
  // ext: edge = ck[1] != ck[2]; falling (pdm_edge 0) or rising (1). Take the data sampled with
  // the last pre-edge clock sample (sync[2]).
  wire ext_sample = pdm_edge ? (ck[1] & ~ck[2]) : (~ck[1] & ck[2]);
  wire sample_now = ext_clk ? ext_sample : int_sample;
  wire bit_in = ext_clk ? sync[2] : sync[1];

  // Integrators run at the PDM rate. Input bit maps to +1 / -1.
  reg signed [W-1:0] i1, i2, i3;
  wire signed [W-1:0] x = bit_in ? {{(W - 1) {1'b0}}, 1'b1} : {W{1'b1}};

  reg [7:0] dec_cnt;
  wire      dec_last = (dec_cnt == decim - 8'd1);

  // Comb, one stage per clock: stage 1 latches i3 (one clock after the last integrator update,
  // when i3 holds it), then c1, c2, c3, then the scaled, saturated output.
  reg [3:0] comb;  // one-hot pipeline position: [0] snapshot ready, [1] c1, [2] c2, [3] c3
  reg                dec_pending;
  reg signed [W-1:0] snap, c, i3_d, c1_d, c2_d;

  wire signed [W-1:0] c3_s = c >>> shift;
  wire sat_hi = (c3_s > 32767);
  wire sat_lo = (c3_s < -32768);

  always @(posedge clk) begin
    if (!rst_n) begin
      div_cnt     <= 0;
      sync        <= 0;
      ck          <= 0;
      i1          <= 0;
      i2          <= 0;
      i3          <= 0;
      dec_cnt     <= 0;
      dec_pending <= 0;
      comb        <= 0;
      snap        <= 0;
      c           <= 0;
      i3_d        <= 0;
      c1_d        <= 0;
      c2_d        <= 0;
      pcm         <= 0;
      pcm_valid   <= 0;
    end else begin
      div_cnt   <= div_cnt + 1'b1;
      sync      <= {sync[1:0], pdm_data};
      ck        <= {ck[1:0], pdm_clk_in};
      pcm_valid <= 0;
      comb      <= {comb[2:0], 1'b0};

      if (sample_now) begin
        // Pipelined integrators (each uses the previous stage's old value).
        i1      <= i1 + x;
        i2      <= i2 + i1;
        i3      <= i3 + i2;
        dec_cnt <= dec_last ? 8'd0 : dec_cnt + 1'b1;
        if (dec_last) dec_pending <= 1;
      end

      // One cycle after the R-th integrator update, i3 holds the new value.
      if (dec_pending) begin
        dec_pending <= 0;
        snap        <= i3;
        comb[0]     <= 1'b1;
      end
      if (comb[0]) begin  // c1 = i3 - i3_d
        c    <= snap - i3_d;
        i3_d <= snap;
      end
      if (comb[1]) begin  // c2 = c1 - c1_d
        c    <= c - c1_d;
        c1_d <= c;
      end
      if (comb[2]) begin  // c3 = c2 - c2_d
        c    <= c - c2_d;
        c2_d <= c;
      end
      if (comb[3]) begin
        pcm       <= sat_hi ? 16'h7fff : sat_lo ? 16'h8000 : c3_s[15:0];
        pcm_valid <= 1;
      end
    end
  end

endmodule
