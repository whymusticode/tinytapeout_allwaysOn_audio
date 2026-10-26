/*
 * Copyright (c) 2026 Michael Benton
 * SPDX-License-Identifier: Apache-2.0
 *
 * Always-on wake-word detector, top level. Pin muxing only.
 *
 * Pin map (see info.yaml / docs/info.md):
 *   ui[0]   PDM_DATA      mic data in
 *   ui[1]   PDM_EDGE      0: data valid while PDM_CLK high, 1: while low
 *   ui[2]   SPI_SCK       host SPI (reserved)
 *   ui[3]   SPI_CS_N      host SPI (reserved)
 *   ui[4]   SPI_MOSI      host SPI (reserved)
 *   ui[7:5] VAD_THR       bring-up VAD threshold, x512 (until SPI config exists)
 *   uo[0]   WAKE_IRQ      wake word detected (not yet implemented)
 *   uo[1]   PDM_CLK       mic clock, clk/16 (1.024 MHz at 16.384 MHz)
 *   uo[2]   SPI_MISO      host SPI (reserved)
 *   uo[3]   VAD           voice activity
 *   uio     QSPI Pmod     weight flash / PSRAM (reserved, TT qspi-pmod pinout)
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

  assign uo_out  = {4'b0, vad_active, 1'b0, pdm_clk, 1'b0};
  assign uio_out = 0;
  assign uio_oe  = 0;

  // List all unused inputs to prevent warnings
  wire _unused = &{ena, ui_in[4:2], uio_in, vad_env, 1'b0};

endmodule
