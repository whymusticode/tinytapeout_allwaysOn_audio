/*
 * pdm_capture: clock two PDM mics from the Raspberry Pi 5's RP1 PIO and stream the raw
 * samples to stdout (decoded by pdm.py / record.py).
 *
 *   CLK  GPIO5  -> both mics' CLK, 1.024 MHz (the chip's PDM_CLK: 16.384 MHz / 16)
 *   DAT0 GPIO6  <- mic 0 DAT
 *   DAT1 GPIO7  <- mic 1 DAT
 *
 * The state machine samples both data pins at the end of each half of the clock, so it
 * captures a mic whichever half its SEL pin puts it on (and a shared-line stereo pair too).
 * Output: 32-bit words, 16 two-bit groups each, MSB first in time; groups alternate
 * low-half / high-half; bit 0 = DAT0, bit 1 = DAT1. 4 bits per PDM clock = 512 kB/s.
 *
 * Program (side-set 1 = CLK), 6 PIO cycles per PDM clock:
 *   0: nop        side 0 [1]
 *   1: in pins, 2 side 0        ; end of low half
 *   2: nop        side 1 [1]
 *   3: in pins, 2 side 1        ; end of high half
 *
 * Build: setup.sh (needs piolib from github.com/raspberrypi/utils).
 */

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "piolib.h"

#define PIN_CLK 5
#define PIN_DAT0 6 /* DAT1 = PIN_DAT0 + 1: `in pins, 2` reads consecutive pins */
#define PDM_HZ 1024000
#define CYCLES_PER_CLK 6
#define BUF_BYTES 32768

static const uint16_t prog_instr[] = {
    0xA142, /* nop        side 0 [1]  (mov y, y) */
    0x4002, /* in pins, 2 side 0 */
    0xB142, /* nop        side 1 [1] */
    0x5002, /* in pins, 2 side 1 */
};
static const pio_program_t prog = {
    .instructions = prog_instr,
    .length = 4,
    .origin = -1,
};

static volatile sig_atomic_t stop;
static void on_signal(int s) { (void)s; stop = 1; }

int main(void) {
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, on_signal);

    PIO pio = pio0;
    if (PIO_IS_ERR(pio)) {
        fprintf(stderr, "pdm_capture: can't open PIO (is /dev/pio0 there? try sudo)\n");
        return 1;
    }
    int sm = pio_claim_unused_sm(pio, true);
    uint offset = pio_add_program(pio, &prog);

    pio_sm_config c = pio_get_default_sm_config();
    sm_config_set_wrap(&c, offset, offset + 3);
    sm_config_set_sideset(&c, 1, false, false);
    sm_config_set_sideset_pins(&c, PIN_CLK);
    sm_config_set_in_pins(&c, PIN_DAT0);
    sm_config_set_in_shift(&c, false, true, 32); /* shift left, autopush every 32 bits */
    sm_config_set_fifo_join(&c, PIO_FIFO_JOIN_RX);
    float div = (float)clock_get_hz(clk_sys) / (PDM_HZ * CYCLES_PER_CLK);
    sm_config_set_clkdiv(&c, div);

    pio_gpio_init(pio, PIN_CLK);
    pio_gpio_init(pio, PIN_DAT0);
    pio_gpio_init(pio, PIN_DAT0 + 1);
    pio_sm_set_consecutive_pindirs(pio, sm, PIN_CLK, 1, true);
    pio_sm_set_consecutive_pindirs(pio, sm, PIN_DAT0, 2, false);
    pio_sm_init(pio, sm, offset, &c);
    pio_sm_config_xfer(pio, sm, PIO_DIR_FROM_SM, BUF_BYTES, 4);
    fprintf(stderr, "pdm_capture: CLK GPIO%d at %d Hz (PIO clock %u Hz, div %.4f), DAT GPIO%d/%d\n",
            PIN_CLK, PDM_HZ, (unsigned)clock_get_hz(clk_sys), div, PIN_DAT0, PIN_DAT0 + 1);
    pio_sm_set_enabled(pio, sm, true);

    uint32_t *buf = malloc(BUF_BYTES);
    while (!stop) {
        if (pio_sm_xfer_data(pio, sm, PIO_DIR_FROM_SM, BUF_BYTES, buf) < 0) {
            fprintf(stderr, "pdm_capture: transfer failed\n");
            break;
        }
        if (fwrite(buf, 1, BUF_BYTES, stdout) != BUF_BYTES)
            break;
    }
    pio_sm_set_enabled(pio, sm, false);
    free(buf);
    return 0;
}
