/*
 * Copyright (c) 2026 Michael Benton
 * SPDX-License-Identifier: Apache-2.0
 *
 * Always-on wake-word detector, top level. Pin muxing only.
 *
 * Pin map (see info.yaml / docs/info.md):
 *   ui[0]   PDM_DATA      mic data in
 *   ui[1]   PDM_EDGE      0: data valid while PDM_CLK high, 1: while low
 *   ui[2]   STREAM_REQ    host holds high: stream the last 2 s of audio, then live audio, on AUDIO_TX
 *   ui[3]   SPI_CS_N      host SPI (reserved)
 *   ui[4]   SPI_MOSI      host SPI (reserved)
 *   ui[7:5] VAD_THR       bring-up VAD threshold, x512 (until SPI config exists)
 *   uo[0]   WAKE_IRQ      wake word detected (not yet implemented)
 *   uo[1]   PDM_CLK       mic clock, clk/16 (1.024 MHz at 16.384 MHz)
 *   uo[2]   AUDIO_TX      UART 8N1, 1.024 Mbaud: "WAKE" then int16 little-endian PCM at 16 kHz
 *   uo[3]   VAD           voice activity
 *   uo[4]   STREAMING     a stream session is active
 *   uio     QSPI Pmod     TT qspi-pmod pinout: uio[6] PSRAM A = rolling audio buffer (SPI mode),
 *                         uio[0] flash and uio[7] PSRAM B deselected (weights later)
 */

`default_nettype none

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

  wire        pdm_clk;
  wire [15:0] pcm;
  wire        pcm_valid;
  wire        vad_active;
  wire [14:0] vad_env;

  pdm_cic u_cic (
      .clk      (clk),
      .rst_n    (rst_n),
      .pdm_data (ui_in[0]),
      .pdm_edge (ui_in[1]),
      .pdm_clk  (pdm_clk),
      .pcm      (pcm),
      .pcm_valid(pcm_valid)
  );

  vad u_vad (
      .clk      (clk),
      .rst_n    (rst_n),
      .pcm      (pcm),
      .pcm_valid(pcm_valid),
      .threshold({3'b0, ui_in[7:5], 9'd0}),
      .active   (vad_active),
      .envelope (vad_env)
  );

  // Rolling audio buffer in PSRAM A, streamed to the host on request (second-stage verifier).
  wire audio_tx, streaming, ram_cs_n, spi_sck, spi_mosi;

  audio_ring u_ring (
      .clk       (clk),
      .rst_n     (rst_n),
      .pcm       (pcm),
      .pcm_valid (pcm_valid),
      .stream_req(ui_in[2]),
      .uart_tx   (audio_tx),
      .streaming (streaming),
      .ram_cs_n  (ram_cs_n),
      .spi_sck   (spi_sck),
      .spi_mosi  (spi_mosi),
      .spi_miso  (uio_in[2])
  );

  assign uo_out = {3'b0, streaming, vad_active, audio_tx, pdm_clk, 1'b0};

  // QSPI Pmod: [0] flash CS, [1] SD0/MOSI, [2] SD1/MISO, [3] SCK, [4] SD2, [5] SD3, [6] RAM A CS, [7] RAM B CS.
  // SD2/SD3 are held high (flash WP#/HOLD# inactive) while the bus runs in plain SPI mode.
  assign uio_out = {1'b1, ram_cs_n, 1'b1, 1'b1, spi_sck, 1'b0, spi_mosi, 1'b1};
  assign uio_oe  = 8'b1111_1011;

  // List all unused inputs to prevent warnings
  wire _unused = &{ena, ui_in[4:3], uio_in[7:3], uio_in[1:0], vad_env, 1'b0};

endmodule
