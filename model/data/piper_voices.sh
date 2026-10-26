#!/usr/bin/env bash
# Piper TTS voices (English) for the simulator: data/piper/<voice>.onnx + .onnx.json
# Multi-speaker models give many speakers; single-speaker models add distinct styles/accents.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p data/piper
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main
VOICES="
en_GB-vctk-medium en_US-arctic-medium en_US-l2arctic-medium en_US-libritts_r-medium
en_US-libritts-high en_GB-aru-medium en_GB-semaine-medium
en_GB-alan-medium en_GB-alba-medium en_GB-cori-medium en_GB-jenny_dioco-medium
en_GB-northern_english_male-medium en_GB-southern_english_female-low
en_US-amy-medium en_US-bryce-medium en_US-danny-low en_US-hfc_female-medium en_US-hfc_male-medium
en_US-joe-medium en_US-john-medium en_US-kathleen-low en_US-kristin-medium en_US-kusal-medium
en_US-lessac-medium en_US-ljspeech-medium en_US-mike-medium en_US-norman-medium
en_US-reza_ibrahim-medium en_US-ryan-medium en_US-sam-medium
"
for v in $VOICES; do
  [ -s data/piper/$v.onnx ] && continue
  IFS=- read -r loc name quality <<< "$v"
  lang=${loc%%_*}
  for ext in onnx onnx.json; do
    curl -fsSL --retry 3 -o data/piper/$v.$ext.part "$BASE/$lang/$loc/$name/$quality/$v.$ext" && mv data/piper/$v.$ext.part data/piper/$v.$ext
  done
  echo "got $v"
done
