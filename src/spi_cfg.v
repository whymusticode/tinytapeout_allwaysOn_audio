/*
 * Copyright (c) 2026 Michael Benton
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host SPI: configuration registers, status/debug readout, and PCM injection.
 *
 * SPI mode 0, MSB first, SCK at most clk / 16 (1 MHz at 16.384 MHz): the pins are synchronised
 * and edge-detected in the clk domain, and MISO changes ~3 clocks after SCK falls.
 * Transaction: CS low, command byte {read, addr[6:0]}, then data bytes; the address increments
 * after every data byte. On a read the first data byte is register `addr`.
 *
 *   addr  name      reset  access
 *   0x00  ID        0x57   r     'W'
 *   0x01  CTRL      0x01   rw    [0] miso_late: PSRAM MISO sampled at SCK fall (2 clk round trip)
 *                                [1] inject: PCM comes from INJ_* writes, not the mic
 *   0x02  VAD_THR0  0x00   rw    VAD threshold [7:0]
 *   0x03  VAD_THR1  0x04   rw    VAD threshold [14:8] (reset 1024)
 *   0x04  DECIM     64     rw    CIC decimation R (PDM rate / 16 kHz)
 *   0x05  SHIFT     3      rw    CIC output shift [3:0]
 *   0x06  DBG_SEL   0      rw    debug pins uo[7:6]: 0 {pcm_valid, vad}, 1 {ram_cs_n, ram_miso},
 *                                2 {feat_valid, fb_busy}, 3 {injected sample strobe, mic PCM strobe}
 *   0x08  INJ0      0      w     injected PCM [7:0]
 *   0x09  INJ1      0      w     injected PCM [15:8]; the write issues the sample
 *   0x10  STATUS    -      r     [0] vad [1] streaming [2] ext_clk strap [3] pdm_edge strap
 *                                [4] inject overrun (sticky, cleared by reading STATUS)
 *   0x11  ENV0/1    -      r     VAD envelope [7:0], 0x12 [14:8]
 *   0x13  PCM0/1    -      r     last PCM sample; reading 0x13 latches the high byte for 0x14
 *   0x15  PCM_CNT   -      r     PCM samples seen, mod 256 (is the mic alive?)
 *   0x16  FEAT_CNT  -      r     filterbank frames, mod 256
 *   0x20+b FEAT[b]  -      r     band b's code in the latest filterbank frame (ext_rdata)
 *
 * Injection: the host writes INJ0, INJ1 per sample, at most one sample per 1024 clocks (16 kHz),
 * the front end's real-time rate; a sample arriving while the filterbank is still busy is
 * dropped and sets the overrun bit.
 */

`default_nettype none

module spi_cfg (
    input  wire        clk,
    input  wire        rst_n,
    // pins (asynchronous)
    input  wire        cs_n,
    input  wire        sck,
    input  wire        mosi,
    output wire        miso,
    // configuration
    output wire        miso_late,
    output wire        inject,
    output wire [14:0] vad_thr,
    output wire [ 7:0] decim,
    output wire [ 3:0] shift,
    output wire [ 1:0] dbg_sel,
    // injected PCM
    output reg  [15:0] inj_pcm,
    output reg         inj_valid,
    input  wire        fb_busy,
    // status
    input  wire        vad,
    input  wire        streaming,
    input  wire        ext_clk,
    input  wire        pdm_edge,
    input  wire [14:0] vad_env,
    input  wire [15:0] pcm,
    input  wire        pcm_valid,
    input  wire        feat_frame,
    // registers 0x20-0x3F, held outside: ext_addr = addr[4:0] of the byte being loaded
    output wire [ 4:0] ext_addr,
    input  wire [ 7:0] ext_rdata
);

  reg [1:0] cs_s, mosi_s;
  reg [2:0] sck_s;
  wire sel = !cs_s[1];
  wire rise = sck_s[1] & ~sck_s[2];
  wire fall = ~sck_s[1] & sck_s[2];

  reg [7:0] ctrl, thr0, thr1, r_decim, r_shift, r_dbg, inj_lo;
  reg       overrun;
  reg [7:0] pcm_cnt, feat_cnt, pcm_hi_latch;

  assign miso_late = ctrl[0];
  assign inject    = ctrl[1];
  assign vad_thr   = {thr1[6:0], thr0};
  assign decim     = r_decim;
  assign shift     = r_shift[3:0];
  assign dbg_sel   = r_dbg[1:0];

  reg [2:0] bitcnt;
  reg [6:0] rx;
  reg [7:0] tx;
  reg       first, rd;
  reg [6:0] addr;
  wire [7:0] rx_next = {rx[6:0], mosi_s[1]};
  wire [6:0] rd_a = first ? rx_next[6:0] : addr;  // register a read loads at this byte's end
  assign ext_addr = rd_a[4:0];
  wire _unused = &{1'b0, rd_a[6:5]};

  function automatic [7:0] rdata(input [6:0] a);
    case (a)
      7'h00: rdata = 8'h57;
      7'h01: rdata = ctrl;
      7'h02: rdata = thr0;
      7'h03: rdata = thr1;
      7'h04: rdata = r_decim;
      7'h05: rdata = r_shift;
      7'h06: rdata = r_dbg;
      7'h10: rdata = {3'b0, overrun, pdm_edge, ext_clk, streaming, vad};
      7'h11: rdata = vad_env[7:0];
      7'h12: rdata = {1'b0, vad_env[14:8]};
      7'h13: rdata = pcm[7:0];
      7'h14: rdata = pcm_hi_latch;
      7'h15: rdata = pcm_cnt;
      7'h16: rdata = feat_cnt;
      default: rdata = (a[6:5] == 2'b01) ? ext_rdata : 8'h00;
    endcase
  endfunction

  assign miso = sel & tx[7];

  always @(posedge clk) begin
    if (!rst_n) begin
      cs_s         <= 2'b11;
      sck_s        <= 0;
      mosi_s       <= 0;
      ctrl         <= 8'h01;
      thr0         <= 8'h00;
      thr1         <= 8'h04;
      r_decim      <= 8'd64;
      r_shift      <= 8'd3;
      r_dbg        <= 0;
      inj_lo       <= 0;
      inj_pcm      <= 0;
      inj_valid    <= 0;
      overrun      <= 0;
      pcm_cnt      <= 0;
      feat_cnt     <= 0;
      pcm_hi_latch <= 0;
      bitcnt       <= 0;
      rx           <= 0;
      tx           <= 0;
      first        <= 1;
      rd           <= 0;
      addr         <= 0;
    end else begin
      cs_s      <= {cs_s[0], cs_n};
      sck_s     <= {sck_s[1:0], sck};
      mosi_s    <= {mosi_s[0], mosi};
      inj_valid <= 0;
      if (pcm_valid) pcm_cnt <= pcm_cnt + 1'b1;
      if (feat_frame) feat_cnt <= feat_cnt + 1'b1;

      if (!sel) begin
        bitcnt <= 0;
        first  <= 1;
        tx     <= 0;
      end else if (rise) begin
        rx     <= rx_next[6:0];
        bitcnt <= bitcnt + 1'b1;
        if (bitcnt == 3'd7) begin
          if (first) begin
            first <= 0;
            rd    <= rx_next[7];
            addr  <= rx_next[6:0] + (rx_next[7] ? 7'd1 : 7'd0);
            if (rx_next[7]) begin
              tx <= rdata(rx_next[6:0]);
              if (rx_next[6:0] == 7'h13) pcm_hi_latch <= pcm[15:8];
              if (rx_next[6:0] == 7'h10) overrun <= 0;
            end
          end else begin
            addr <= addr + 1'b1;
            if (rd) begin
              tx <= rdata(addr);
              if (addr == 7'h13) pcm_hi_latch <= pcm[15:8];
              if (addr == 7'h10) overrun <= 0;
            end else begin
              case (addr)
                7'h01: ctrl <= rx_next;
                7'h02: thr0 <= rx_next;
                7'h03: thr1 <= rx_next;
                7'h04: r_decim <= rx_next;
                7'h05: r_shift <= rx_next;
                7'h06: r_dbg <= rx_next;
                7'h08: inj_lo <= rx_next;
                7'h09: begin
                  if (fb_busy) overrun <= 1;
                  else begin
                    inj_pcm   <= {rx_next, inj_lo};
                    inj_valid <= 1;
                  end
                end
                default: ;
              endcase
            end
          end
        end
      end else if (fall && bitcnt != 0) begin
        tx <= {tx[6:0], 1'b0};
      end
    end
  end

endmodule
