/*
 * Copyright (c) 2026 Michael Benton
 * SPDX-License-Identifier: Apache-2.0
 *
 * Rolling audio buffer for the second-stage host.
 *
 * Every 16 kHz PCM sample is written to a ring buffer in the QSPI Pmod's PSRAM A
 * (APS6404L, plain SPI mode), 2^BUF_LOG2 samples deep (2^16 = 4.1 s). While the host holds
 * STREAM_REQ high, the buffer is streamed out on a UART (8N1, clk / CLKS_PER_BAUD =
 * 1.024 Mbaud): first the magic bytes "WAKE", then 16-bit little-endian samples starting
 * 2^PRE_LOG2 samples (2^15 = 2.05 s) before the request, continuing with live audio until
 * STREAM_REQ drops. The UART is ~3x faster than the audio, so the stream catches up with
 * real time. This is the cascade pattern of the big players: the always-on stage keeps a
 * short history (Google: 2 s; Amazon: 0.5 s pre-roll + keyword + following speech) for a
 * larger verifier.
 *
 * PSRAM traffic: one SPI transaction per sample written and per sample read,
 * {cmd 8, addr 24, data 16} = 48 SCK cycles at clk/2 = 5.9 us, inside the APS6404L's 8 us
 * maximum CE# low time (needed for its refresh). Writes have priority over reads.
 * Power-up: wait INIT_CYCLES (>150 us), then reset-enable (0x66) and reset (0x99).
 */

`default_nettype none

module audio_ring #(
    parameter BUF_LOG2      = 16,    // ring size, samples
    parameter PRE_LOG2      = 15,    // pre-roll, samples
    parameter CLKS_PER_BAUD = 16,    // 16.384 MHz / 16 = 1.024 Mbaud
    parameter INIT_CYCLES   = 4096   // 250 us at 16.384 MHz
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] pcm,
    input  wire        pcm_valid,
    input  wire        stream_req,   // asynchronous, from a pin
    output wire        uart_tx,
    output wire        streaming,
    // SPI to PSRAM (mode 0)
    output reg         ram_cs_n,
    output reg         spi_sck,
    output wire        spi_mosi,
    input  wire        spi_miso
);

  // ---------------- SPI transaction engine ----------------
  localparam [7:0] CMD_WRITE = 8'h02, CMD_READ = 8'h03, CMD_RSTEN = 8'h66, CMD_RST = 8'h99;

  reg  [47:0] sr;        // bits shifted out, MSB first
  reg  [15:0] rx;        // bits shifted in
  reg  [ 5:0] bits_left; // bits still to send in this transaction
  reg         busy;
  wire        spi_done;  // one-cycle pulse when a transaction ends

  assign spi_mosi = sr[47];

  reg done_q;
  assign spi_done = done_q;

  // ---------------- control ----------------
  localparam [1:0] I_WAIT = 2'd0, I_RSTEN = 2'd1, I_RST = 2'd2, I_DONE = 2'd3;
  reg [ 1:0] init_state;
  reg [12:0] init_cnt;

  reg [BUF_LOG2-1:0] wr_ptr, rd_ptr;
  reg [15:0] wbuf;
  reg        wpend;
  reg        cur_write;  // the running transaction is a sample write

  reg [1:0] req_sync;
  reg       req_q;
  reg       active;      // streaming session
  assign streaming = active;

  reg [15:0] hold;       // sample waiting for the UART
  reg [ 1:0] hold_n;     // bytes of `hold` still to send
  reg [ 2:0] hdr_n;      // header bytes still to send
  reg        rd_pend;    // a read transaction is running

  // UART
  reg [ 9:0] tx_sr;      // {stop, data[7:0], start}, LSB first
  reg [ 3:0] tx_bits;
  reg [ 4:0] tx_div;
  wire       tx_idle = (tx_bits == 0);
  assign uart_tx = tx_idle ? 1'b1 : tx_sr[0];

  wire [23:0] wr_addr = {{(23 - BUF_LOG2) {1'b0}}, wr_ptr, 1'b0};
  wire [23:0] rd_addr = {{(23 - BUF_LOG2) {1'b0}}, rd_ptr, 1'b0};

  function automatic [7:0] hdr_byte(input [2:0] n);  // "WAKE", sent as n = 4, 3, 2, 1
    case (n)
      3'd4: hdr_byte = 8'h57;
      3'd3: hdr_byte = 8'h41;
      3'd2: hdr_byte = 8'h4B;
      default: hdr_byte = 8'h45;
    endcase
  endfunction

  always @(posedge clk) begin
    if (!rst_n) begin
      ram_cs_n   <= 1'b1;
      spi_sck    <= 1'b0;
      sr         <= 0;
      rx         <= 0;
      bits_left  <= 0;
      busy       <= 1'b0;
      done_q     <= 1'b0;
      init_state <= I_WAIT;
      init_cnt   <= 0;
      wr_ptr     <= 0;
      rd_ptr     <= 0;
      wbuf       <= 0;
      wpend      <= 1'b0;
      cur_write  <= 1'b0;
      req_sync   <= 0;
      req_q      <= 1'b0;
      active     <= 1'b0;
      hold       <= 0;
      hold_n     <= 0;
      hdr_n      <= 0;
      rd_pend    <= 1'b0;
      tx_sr      <= 10'h3ff;
      tx_bits    <= 0;
      tx_div     <= 0;
    end else begin
      done_q <= 1'b0;

      // ---- SPI bit engine: 2 clocks per bit, MOSI changes while SCK is low ----
      if (busy) begin
        if (!spi_sck) begin
          spi_sck <= 1'b1;
          rx      <= {rx[14:0], spi_miso};  // sample at the rising edge
        end else begin
          spi_sck   <= 1'b0;
          sr        <= {sr[46:0], 1'b0};
          bits_left <= bits_left - 1'b1;
          if (bits_left == 1) begin
            busy     <= 1'b0;
            ram_cs_n <= 1'b1;
            done_q   <= 1'b1;
          end
        end
      end

      // ---- new samples to write ----
      if (pcm_valid) begin
        wbuf  <= pcm;
        wpend <= 1'b1;
      end

      // ---- stream session control ----
      req_sync <= {req_sync[0], stream_req};
      req_q    <= req_sync[1];
      if (req_sync[1] && !req_q && init_state == I_DONE) begin
        active <= 1'b1;
        rd_ptr <= wr_ptr - (1 << PRE_LOG2);
        hdr_n  <= 3'd4;
        hold_n <= 0;
      end else if (!req_sync[1] && req_q) begin
        active <= 1'b0;
        hdr_n  <= 0;
        hold_n <= 0;
      end

      // ---- transaction completion ----
      if (spi_done) begin
        if (cur_write) begin
          wr_ptr    <= wr_ptr + 1'b1;
          cur_write <= 1'b0;
        end
        if (rd_pend) begin
          rd_pend <= 1'b0;
          if (active) begin
            hold   <= rx;
            hold_n <= 2'd2;
            rd_ptr <= rd_ptr + 1'b1;
          end
        end
      end

      // ---- issue the next transaction (init, then writes before reads) ----
      if (!busy && !spi_done && ram_cs_n) begin
        case (init_state)
          I_WAIT: begin
            init_cnt <= init_cnt + 1'b1;
            if (init_cnt == INIT_CYCLES) init_state <= I_RSTEN;
          end
          I_RSTEN, I_RST: begin
            sr         <= {(init_state == I_RSTEN) ? CMD_RSTEN : CMD_RST, 40'd0};
            bits_left  <= 6'd8;
            busy       <= 1'b1;
            ram_cs_n   <= 1'b0;
            init_state <= (init_state == I_RSTEN) ? I_RST : I_DONE;
          end
          default: begin
            if (wpend && !pcm_valid) begin
              sr        <= {CMD_WRITE, wr_addr, wbuf};
              bits_left <= 6'd48;
              busy      <= 1'b1;
              ram_cs_n  <= 1'b0;
              wpend     <= 1'b0;
              cur_write <= 1'b1;
            end else if (active && hold_n == 0 && !rd_pend && rd_ptr != wr_ptr && !wpend) begin
              sr        <= {CMD_READ, rd_addr, 16'd0};
              bits_left <= 6'd48;
              busy      <= 1'b1;
              ram_cs_n  <= 1'b0;
              rd_pend   <= 1'b1;
            end
          end
        endcase
      end

      // ---- UART: header, then each sample low byte first ----
      if (!tx_idle) begin
        if (tx_div == CLKS_PER_BAUD - 1) begin
          tx_div  <= 0;
          tx_sr   <= {1'b1, tx_sr[9:1]};
          tx_bits <= tx_bits - 1'b1;
        end else begin
          tx_div <= tx_div + 1'b1;
        end
      end else if (active && hdr_n != 0) begin
        tx_sr   <= {1'b1, hdr_byte(hdr_n), 1'b0};
        tx_bits <= 4'd10;
        hdr_n   <= hdr_n - 1'b1;
      end else if (active && hold_n != 0 && !(spi_done && rd_pend)) begin
        tx_sr   <= {1'b1, (hold_n == 2) ? hold[7:0] : hold[15:8], 1'b0};
        tx_bits <= 4'd10;
        hold_n  <= hold_n - 1'b1;
      end
    end
  end

endmodule
