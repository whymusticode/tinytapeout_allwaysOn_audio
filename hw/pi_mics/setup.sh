#!/usr/bin/env bash
# One-time setup on the Raspberry Pi 5 (Raspberry Pi OS Bookworm or later, 64-bit).
#   ./setup.sh            build pdm_capture + make the Python venv
#   ./setup.sh service    also install the always-on recorder as a systemd service
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)

if [ ! -e /dev/pio0 ]; then
  echo "No /dev/pio0: update the OS (sudo apt full-upgrade) so the kernel has the rp1-pio driver." >&2
  exit 1
fi

sudo apt-get install -y git cmake build-essential python3-venv libsndfile1

# piolib: user-space access to the RP1's PIO blocks
if [ ! -d "$HOME/rpi-utils" ]; then
  git clone --depth 1 https://github.com/raspberrypi/utils.git "$HOME/rpi-utils"
fi
cmake -S "$HOME/rpi-utils/piolib" -B "$HOME/rpi-utils/build-piolib" -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/rpi-utils/build-piolib" -j4
LIB=$(find "$HOME/rpi-utils/build-piolib" -name 'libpio*.a' -o -name 'libpio*.so' | head -n1)
INC="$HOME/rpi-utils/piolib/include"
echo "piolib: $LIB"
gcc -O2 -Wall -o pdm_capture pdm_capture.c -I"$INC" "$LIB"
echo "built $HERE/pdm_capture"

# Python env (uv if available, else venv + pip)
if command -v uv >/dev/null; then
  uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
else
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
fi

if [ "${1:-}" = "service" ]; then
  sed -e "s|@HERE@|$HERE|g" -e "s|@OUT@|$HOME/recordings|g" pdm-record.service | sudo tee /etc/systemd/system/pdm-record.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now pdm-record
  echo "recorder running: journalctl -u pdm-record -f"
fi
