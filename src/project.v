/*
 * Copyright (c) 2026 Michael Benton
 * SPDX-License-Identifier: Apache-2.0
 *
 * Always-on wake-word detector, top level.
 *
 * Pin map (see info.yaml / docs/info.md):
 *   ui[0]   PDM_DATA      mic data in
 *   ui[1]   PDM_CLK_IN    mic clock in, when someone else drives the mic (PDM_EXT = 1)
 *   ui[2]   STREAM_REQ    host holds high: stream the last 2 s of audio, then live audio, on AUDIO_TX
 *   ui[3]   SPI_CS_N      host SPI: configuration, status, feature readout, PCM injection (spi_cfg.v)
 *   ui[4]   SPI_SCK
 *   ui[5]   SPI_MOSI
 *   ui[6]   PDM_EXT       strap: 0 = we clock the mic (uo[1]), 1 = the mic clock comes in on ui[1]
 *                         (listening in on a phone's mic, clocked by its codec)
 *   ui[7]   PDM_EDGE      strap: 0 = data valid while the mic clock is high, 1 = while low
 *   uo[0]   WAKE_IRQ      wake word detected (not yet implemented)
 *   uo[1]   PDM_CLK       mic clock, clk/16 (1.024 MHz at 16.384 MHz); low when PDM_EXT = 1
 *   uo[2]   AUDIO_TX      UART 8N1, 1.024 Mbaud: "WAKE" then int16 little-endian PCM at 16 kHz
 *   uo[3]   VAD           voice activity
 *   uo[4]   STREAMING     a stream session is active
 *   uo[5]   SPI_MISO
 *   uo[7:6] DEBUG         two internal signals, chosen by the DBG_SEL register
 *   uio     QSPI Pmod     TT qspi-pmod pinout: uio[6] PSRAM A = rolling audio buffer (SPI mode),
 *                         uio[0] flash and uio[7] PSRAM B deselected (weights later)
 *
 * Signal path: mic -> pdm_cic -> (or host-injected PCM) -> vad, audio_ring, filterbank -> feature
 * frame (readable over SPI; the engine's input later).
 */

`default_nettype none

/* verilator lint_off DECLFILENAME */  // TT's naming: project.v holds tt_um_<name>
module tt_um_whymusticode_wakeword (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

  localparam N_BANDS = 12;  // Stage A's front end (search_space.FRONTEND_DEFAULT); 8..24
  localparam FEAT_BITS = 6;

  // rst_n is an asynchronous pin; every flop uses a synchronous reset, so release it on a
  // clock edge, the same one for all of them.
  reg [1:0] rst_sync;
  always @(posedge clk) rst_sync <= {rst_sync[0], rst_n};
  wire rst_n_s = rst_sync[1];

  wire ext_clk = ui_in[6];
  wire pdm_edge = ui_in[7];

  // ---- configuration ----
  wire        miso_late, inject, spi_miso;
  wire [14:0] vad_thr;
  wire [ 7:0] decim;
  wire [ 3:0] shift;
  wire [ 1:0] dbg_sel;
  wire [15:0] inj_pcm;
  wire        inj_valid;

  // ---- mic -> PCM ----
  wire        pdm_clk;
  wire [15:0] mic_pcm;
  wire        mic_valid;

  pdm_cic u_cic (
      .clk       (clk),
      .rst_n     (rst_n_s),
      .pdm_data  (ui_in[0]),
      .pdm_edge  (pdm_edge),
      .ext_clk   (ext_clk),
      .pdm_clk_in(ui_in[1]),
      .decim     (decim),
      .shift     (shift),
      .pdm_clk   (pdm_clk),
      .pcm       (mic_pcm),
      .pcm_valid (mic_valid)
  );

  // Host-injected PCM replaces the mic (bring-up / debug: bypasses mic and CIC).
  wire [15:0] pcm = inject ? inj_pcm : mic_pcm;
  wire pcm_valid = inject ? inj_valid : mic_valid;

  wire        vad_active;
  wire [14:0] vad_env;

  vad u_vad (
      .clk      (clk),
      .rst_n    (rst_n_s),
      .pcm      (pcm),
      .pcm_valid(pcm_valid),
      .threshold(vad_thr),
      .active   (vad_active),
      .envelope (vad_env)
  );

  // Rolling audio buffer in PSRAM A, streamed to the host on request (second-stage verifier).
  wire audio_tx, streaming, ram_cs_n, ram_sck, ram_mosi;

  audio_ring u_ring (
      .clk       (clk),
      .rst_n     (rst_n_s),
      .pcm       (pcm),
      .pcm_valid (pcm_valid),
      .stream_req(ui_in[2]),
      .miso_late (miso_late),
      .uart_tx   (audio_tx),
      .streaming (streaming),
      .ram_cs_n  (ram_cs_n),
      .spi_sck   (ram_sck),
      .spi_mosi  (ram_mosi),
      .spi_miso  (uio_in[2])
  );

  // ---- features ----
  wire [FEAT_BITS-1:0] feat;
  wire [4:0] feat_band;
  wire feat_valid, fb_busy;

  filterbank #(.N_BANDS(N_BANDS)) u_fb (
      .clk       (clk),
      .rst_n     (rst_n_s),
      .pcm       (pcm),
      .pcm_valid (pcm_valid),
      .feat      (feat),
      .feat_band (feat_band),
      .feat_valid(feat_valid),
      .busy      (fb_busy)
  );

  // Latest frame, one code per band (the engine's input; readable over SPI at 0x20 + band).
  reg [FEAT_BITS-1:0] frame[0:N_BANDS-1];
  integer b, rb;  // separate loop variables: one is used in a clocked block, one in a combinational
  always @(posedge clk) begin
    if (!rst_n_s) begin
      for (b = 0; b < N_BANDS; b = b + 1) frame[b] <= 0;
    end else if (feat_valid) begin
      for (b = 0; b < N_BANDS; b = b + 1) if (feat_band == b[4:0]) frame[b] <= feat;
    end
  end
  wire frame_done = feat_valid && feat_band == N_BANDS - 1;

  wire [4:0] ext_addr;
  reg  [7:0] ext_rdata;
  always @(*) begin
    ext_rdata = 8'h00;
    for (rb = 0; rb < N_BANDS; rb = rb + 1) if (ext_addr == rb[4:0]) ext_rdata = {{(8 - FEAT_BITS) {1'b0}}, frame[rb]};
  end

  spi_cfg u_cfg (
      .clk       (clk),
      .rst_n     (rst_n_s),
      .cs_n      (ui_in[3]),
      .sck       (ui_in[4]),
      .mosi      (ui_in[5]),
      .miso      (spi_miso),
      .miso_late (miso_late),
      .inject    (inject),
      .vad_thr   (vad_thr),
      .decim     (decim),
      .shift     (shift),
      .dbg_sel   (dbg_sel),
      .inj_pcm   (inj_pcm),
      .inj_valid (inj_valid),
      .fb_busy   (fb_busy),
      .vad       (vad_active),
      .streaming (streaming),
      .ext_clk   (ext_clk),
      .pdm_edge  (pdm_edge),
      .vad_env   (vad_env),
      .pcm       (pcm),
      .pcm_valid (pcm_valid),
      .feat_frame(frame_done),
      .ext_addr  (ext_addr),
      .ext_rdata (ext_rdata)
  );

  reg [1:0] dbg;
  always @(*) begin
    case (dbg_sel)
      2'd0:    dbg = {pcm_valid, vad_active};
      2'd1:    dbg = {ram_cs_n, uio_in[2]};
      2'd2:    dbg = {feat_valid, fb_busy};
      default: dbg = {inj_valid, mic_valid};
    endcase
  end

  assign uo_out = {dbg, spi_miso, streaming, vad_active, audio_tx, pdm_clk, 1'b0};

  // QSPI Pmod: [0] flash CS, [1] SD0/MOSI, [2] SD1/MISO, [3] SCK, [4] SD2, [5] SD3, [6] RAM A CS, [7] RAM B CS.
  // SD2/SD3 are held high (flash WP#/HOLD# inactive) while the bus runs in plain SPI mode.
  assign uio_out = {1'b1, ram_cs_n, 1'b1, 1'b1, ram_sck, 1'b0, ram_mosi, 1'b1};
  assign uio_oe  = 8'b1111_1011;

  // List all unused inputs to prevent warnings
  wire _unused = &{ena, uio_in[7:3], uio_in[1:0], 1'b0};

endmodule
/* verilator lint_on DECLFILENAME */
