/*
 * Copyright (c) 2026 Michael Benton
 * SPDX-License-Identifier: Apache-2.0
 *
 * Energy-envelope voice activity detector (bring-up version).
 *
 *   env += (|pcm| * 16 - env) >>> ENV_SHIFT     (one-pole low pass, 4 fraction bits)
 *   vad  = env / 16 > threshold
 *
 * With ENV_SHIFT = 6 at 16 kHz the time constant is ~4 ms.
 * Bit-exact reference: model/frontend.py:vad()
 */

`default_nettype none

module vad #(
    parameter ENV_SHIFT = 6
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] pcm,
    input  wire        pcm_valid,
    input  wire [14:0] threshold,
    output reg         active,
    output wire [14:0] envelope
);

  // |pcm| saturated to 15 bits (|-32768| -> 32767).
  wire [15:0] mag_full = pcm[15] ? (~pcm + 1'b1) : pcm;
  wire [14:0] mag = mag_full[15] ? 15'h7fff : mag_full[14:0];

  reg  [18:0] env;  // unsigned, 15.4 fixed point
  wire signed [19:0] diff = $signed({1'b0, mag, 4'b0}) - $signed({1'b0, env});
  wire signed [19:0] step = diff >>> ENV_SHIFT;

  assign envelope = env[18:4];
  wire _unused = step[19];  // the sign; env + step stays in 0 .. 2^19 - 1

  always @(posedge clk) begin
    if (!rst_n) begin
      env    <= 0;
      active <= 0;
    end else if (pcm_valid) begin
      env    <= env + step[18:0];
      active <= env[18:4] > threshold;
    end
  end

endmodule
