#!/usr/bin/env bash
# Fetch open datasets for keyword-spotting experiments into $DATA (default ../../data).
# Each dataset is extracted into its own directory under $DATA/raw/<name>/.
set -euo pipefail
DATA=${DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}
OPENSLR=${OPENSLR:-https://openslr.elda.org}  # EU mirror; was ~2x faster than www.openslr.org
ONLY=${ONLY:-}  # optional regex: only fetch datasets whose name matches (lets big ones run in parallel)
mkdir -p "$DATA/dl" "$DATA/raw"

fetch() {  # name url
  local name=$1 url=$2
  local f="$DATA/dl/$(basename "$url")"
  if [ -n "$ONLY" ] && ! [[ $name =~ $ONLY ]]; then return; fi
  if [ -e "$DATA/raw/$name/.done" ]; then echo "[$name] already present"; return; fi
  echo "[$name] downloading $url"
  curl -fL --retry 5 -C - -o "$f" "$url"
  mkdir -p "$DATA/raw/$name"
  case "$f" in
    *.tar.gz|*.tgz) tar -xzf "$f" -C "$DATA/raw/$name" ;;
    *.zip) python3 -I -m zipfile -e "$f" "$DATA/raw/$name" ;;
  esac
  touch "$DATA/raw/$name/.done"
  echo "[$name] done"
}

# Real speakers saying 35 short words (+ _background_noise_), CC-BY 4.0
fetch speech_commands http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
# 2000 environmental sound clips, 50 classes, CC-BY-NC
fetch esc50 https://github.com/karoldvl/ESC-50/archive/master.zip
# Real + simulated room impulse responses and point-source noises
fetch rirs_noises $OPENSLR/resources/28/rirs_noises.zip
# Continuous read speech, used as hard negatives and for false-alarm streams
fetch librispeech_dev $OPENSLR/resources/12/dev-clean.tar.gz
fetch librispeech_test $OPENSLR/resources/12/test-clean.tar.gz
# Music, speech and noise
fetch musan $OPENSLR/resources/17/musan.tar.gz
# Multichannel environmental noise recordings (16 kHz versions)
for env in DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE PCAFETER PRESTO PSTATION SPSQUARE STRAFFIC TBUS TCAR TMETRO; do
  fetch "demand/$env" "https://zenodo.org/records/1227121/files/${env}_16k.zip"
done

# --- Text for the conversation simulator (training uses text only; never real speech audio) ---
# Switchboard transcripts (ISIP word alignments): ~2400 real telephone conversations
fetch text/switchboard https://isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
# DailyDialog: 13k written everyday dialogues (CC BY-NC-SA 4.0)
fetch text/dailydialog https://huggingface.co/datasets/roskoN/dailydialog/resolve/main/train.zip
# OpenSubtitles (OPUS v2018, English mono, 3.6 GB): only the first SUBS_MB compressed MB are kept
SUBS_MB=${SUBS_MB:-400}
if [[ -z $ONLY || text/opensubtitles =~ $ONLY ]] && [ ! -e "$DATA/raw/text/opensubtitles/.done" ]; then
  echo "[text/opensubtitles] streaming first ${SUBS_MB} MB"
  mkdir -p "$DATA/raw/text/opensubtitles"
  { curl -sfL https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz || true; } | head -c "${SUBS_MB}M" |
    { gzip -dc 2>/dev/null || true; } > "$DATA/raw/text/opensubtitles/en.txt"
  touch "$DATA/raw/text/opensubtitles/.done"
  echo "[text/opensubtitles] done"
fi

# --- AMI Meeting Corpus (CC BY 4.0): real multi-party meetings, far-field table mic. TEST ONLY. ---
# ~100 h. Array1-01 is one mic of the table array, i.e. what a device on the table hears.
AMI=https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus
if [[ -z $ONLY || ami =~ $ONLY ]] && [ ! -e "$DATA/raw/ami/.done" ]; then
  mkdir -p "$DATA/raw/ami"
  curl -sfL "$AMI/" | grep -oE 'href="[A-Z]{2}[0-9]{4}[a-z]/' | sed 's/href="//;s/\///' | sort -u |
    xargs -P 6 -I{} sh -c 'f="$1/{}.Array1-01.wav"; [ -s "$f" ] ||
      curl -sfL --retry 5 -C - -o "$f.part" "$2/{}/audio/{}.Array1-01.wav" && mv "$f.part" "$f" ||
      { rm -f "$f.part"; echo "[ami] no array audio for {}"; }' _ "$DATA/raw/ami" "$AMI"
  touch "$DATA/raw/ami/.done"
  echo "[ami] $(ls "$DATA/raw/ami"/*.wav | wc -l) meetings"
fi

# --- MSWC English (CC BY 4.0): real single-word clips from Common Voice. TEST ONLY. ---
# Only the shards holding our candidate words (data/raw/mswc_meta/needed_shards.json, from
# model/data/mswc_shards.sh + the candidate list), and only those words' clips are kept.
MSWC=https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/main/data/opus/en
if [[ -z $ONLY || mswc =~ $ONLY ]] && [ ! -e "$DATA/raw/mswc/.done" ]; then
  mkdir -p "$DATA/raw/mswc"
  words=$(python3 -c "import yaml; print(' '.join(c['phrase'] for c in yaml.safe_load(open('$(dirname "$0")/../wakewords.yaml'))['candidates']))")
  pats=$(for w in $words; do printf "%s_common_voice_* " "$w"; done)
  python3 -c "import json; [print(s, i) for s, i in json.load(open('$DATA/raw/mswc_meta/needed_shards.json'))]" |
    xargs -P 4 -L 1 sh -c 'set -f; out=$1; url=$2; pats=$3; split=$4; i=$5
      mkdir -p "$out/$split"; [ -e "$out/$split/.$i.done" ] && exit 0
      curl -sfL --retry 5 "$url/$split/audio/$i.tar.gz" | tar -xz -C "$out/$split" --wildcards $pats 2>/dev/null
      touch "$out/$split/.$i.done"; echo "[mswc] $split shard $i"' _ "$DATA/raw/mswc" "$MSWC" "$pats"
  touch "$DATA/raw/mswc/.done"
  echo "[mswc] $(find "$DATA/raw/mswc" -name '*.opus' | wc -l) clips"
fi
