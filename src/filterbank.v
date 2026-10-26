/*
 * Copyright (c) 2026 Michael Benton
 * SPDX-License-Identifier: Apache-2.0
 *
 * 16-band mel filterbank -> log2 frame energies.
 *
 * Per PCM sample, for each band b (one shared shift-add multiplier):
 *   v    = (x[n] - x[n-2]) << GUARD
 *   s    = g*v + (-a1)*y1 + (-a2)*y2 + 2^(COEF_FRAC-1)
 *   y    = sat(s >>> COEF_FRAC)
 *   e[b] = sat(e[b] + (|y| >> ACC_SHIFT))
 * Every HOP samples, the 16 log2 codes of e[] are output on feat/feat_band with
 * feat_valid (one band per clock, band 0 first), and e[] is cleared.
 *
 * Per-band state lives in rotating shift registers: bands are processed in order,
 * so the current band is always at index 0 and no read multiplexers are needed.
 *
 * Timing: 3 products x 16 coefficient bits x 16 bands + 2 = 770 cycles per
 * sample, plus 16 cycles every frame for the output. pcm_valid must be at least 800
 * cycles apart (1024 at 16.384 MHz / 16 kHz); `busy` is high while working.
 *
 * Bit-exact reference: model/frontend.py:filterbank()
 */

`default_nettype none

module filterbank #(
    parameter N_BANDS   = 16,
    parameter HOP_LOG2  = 8,   // 256-sample frames (16 ms)
    parameter COEF_FRAC = 14,
    parameter GUARD     = 4,
    parameter Y_W       = 22,
    parameter ACC_SHIFT = 2,
    parameter ACC_W     = 24
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] pcm,
    input  wire        pcm_valid,
    output reg  [ 7:0] feat,
    output reg  [ 3:0] feat_band,
    output reg         feat_valid,
    output wire        busy
);

  `include "filterbank_coeffs.vh"

  localparam V_W = 17 + GUARD;          // (x[n] - x[n-2]) << GUARD
  localparam P_W = 16 + Y_W + 2;        // sum of three 16 x Y_W products, plus rounding

  localparam S_IDLE = 2'd0, S_LOAD = 2'd1, S_MAC = 2'd2, S_OUT = 2'd3;
  reg [1:0] state;
  assign busy = state != S_IDLE;

  // Per-band state, rotated one step after each band: index 0 is the current band.
  reg signed [Y_W-1:0] y1 [0:N_BANDS-1];
  reg signed [Y_W-1:0] y2 [0:N_BANDS-1];
  reg [ACC_W-1:0] e [0:N_BANDS-1];

  reg signed [15:0] xm1, xm2;
  reg signed [V_W-1:0] v;
  reg [HOP_LOG2-1:0] n_in_frame;

  // Multiply-accumulate sequencer: band, term (0: g*v, 1: -a1*y1, 2: -a2*y2), coefficient bit.
  reg [3:0] band;
  reg [1:0] term;
  reg [3:0] bit_i;
  reg signed [P_W-1:0] acc;   // s for the current band
  reg signed [P_W-1:0] dsh;   // data operand << bit_i
  reg [15:0] csh;             // coefficient >> bit_i

  wire signed [15:0] coef = fb_coef(band, term);
  wire signed [Y_W-1:0] y1_b = y1[0];
  wire signed [Y_W-1:0] y2_b = y2[0];

  function automatic signed [P_W-1:0] operand(input [1:0] t);
    case (t)
      2'd0:    operand = {{(P_W - V_W) {v[V_W-1]}}, v};
      2'd1:    operand = {{(P_W - Y_W) {y1_b[Y_W-1]}}, y1_b};
      default: operand = {{(P_W - Y_W) {y2_b[Y_W-1]}}, y2_b};
    endcase
  endfunction

  // Two's complement coefficient: bit 15 has weight -2^15.
  wire signed [P_W-1:0] acc_next = csh[0] ? (bit_i == 4'd15 ? acc - dsh : acc + dsh) : acc;

  // y = sat(acc_next >>> COEF_FRAC), valid on the last bit of term 2.
  wire signed [P_W-COEF_FRAC-1:0] s_sh = acc_next[P_W-1:COEF_FRAC];
  localparam signed [P_W-COEF_FRAC-1:0] Y_MAX = (1 <<< (Y_W - 1)) - 1;
  localparam signed [P_W-COEF_FRAC-1:0] Y_MIN = -(1 <<< (Y_W - 1));
  wire signed [Y_W-1:0] y_new = (s_sh > Y_MAX) ? Y_MAX[Y_W-1:0] :
                                (s_sh < Y_MIN) ? Y_MIN[Y_W-1:0] : s_sh[Y_W-1:0];
  wire [Y_W-1:0] y_abs = y_new[Y_W-1] ? -y_new : y_new;  // |Y_MIN| fits as unsigned
  wire [ACC_W:0] e_sum = {1'b0, e[0]} + (y_abs >> ACC_SHIFT);
  wire [ACC_W-1:0] e_new = e_sum[ACC_W] ? {ACC_W{1'b1}} : e_sum[ACC_W-1:0];

  // log2 code of e[0]: msb position p, next 3 bits m -> {p, m}.
  wire [ACC_W-1:0] e_out = e[0];
  reg [4:0] p;
  integer k;
  always @(*) begin
    p = 0;
    for (k = 0; k < ACC_W; k = k + 1) if (e_out[k]) p = k[4:0];
  end
  wire [ACC_W+2:0] e_norm = {e_out, 3'b0} >> p;
  wire [7:0] log_code = (e_out == 0) ? 8'd0 : {p, e_norm[2:0]};

  localparam signed [P_W-1:0] RND = 1 <<< (COEF_FRAC - 1);

  integer i;
  always @(posedge clk) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      xm1        <= 0;
      xm2        <= 0;
      v          <= 0;
      n_in_frame <= 0;
      band       <= 0;
      term       <= 0;
      bit_i      <= 0;
      acc        <= 0;
      dsh        <= 0;
      csh        <= 0;
      feat       <= 0;
      feat_band  <= 0;
      feat_valid <= 0;
      for (i = 0; i < N_BANDS; i = i + 1) begin
        y1[i] <= 0;
        y2[i] <= 0;
        e[i]  <= 0;
      end
    end else begin
      feat_valid <= 0;
      case (state)
        S_IDLE:
        if (pcm_valid) begin
          v     <= ($signed({pcm[15], pcm}) - $signed({xm2[15], xm2})) <<< GUARD;
          xm2   <= xm1;
          xm1   <= pcm;
          band  <= 0;
          term  <= 0;
          bit_i <= 0;
          acc   <= RND;
          state <= S_LOAD;
        end

        S_LOAD: begin  // v is registered now; load band 0's first operands
          csh   <= fb_coef(4'd0, 2'd0);
          dsh   <= operand(2'd0);
          state <= S_MAC;
        end

        S_MAC: begin
          acc   <= acc_next;
          dsh   <= dsh <<< 1;
          csh   <= csh >> 1;
          bit_i <= bit_i + 1'b1;
          if (bit_i == 4'd15) begin
            if (term != 2'd2) begin
              term <= term + 1'b1;
              csh  <= fb_coef(band, term + 1'b1);
              dsh  <= operand(term + 1'b1);
            end else begin
              for (i = 0; i < N_BANDS - 1; i = i + 1) begin
                y1[i] <= y1[i+1];
                y2[i] <= y2[i+1];
                e[i]  <= e[i+1];
              end
              y2[N_BANDS-1] <= y1_b;
              y1[N_BANDS-1] <= y_new;
              e[N_BANDS-1]  <= e_new;
              term     <= 0;
              acc      <= RND;
              if (band == N_BANDS - 1) begin
                band       <= 0;
                n_in_frame <= n_in_frame + 1'b1;
                state      <= (&n_in_frame) ? S_OUT : S_IDLE;
              end else begin
                band <= band + 1'b1;
                csh  <= fb_coef(band + 1'b1, 2'd0);
                dsh  <= operand(2'd0);
              end
            end
          end
        end

        S_OUT: begin
          feat       <= log_code;
          feat_band  <= band;
          feat_valid <= 1;
          for (i = 0; i < N_BANDS - 1; i = i + 1) e[i] <= e[i+1];
          e[N_BANDS-1] <= 0;
          band       <= band + 1'b1;
          if (band == N_BANDS - 1) state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
