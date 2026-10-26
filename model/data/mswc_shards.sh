#!/usr/bin/env bash
# Print "<split> <shard> <first word>" for every MSWC English opus shard (range request, ~300 KB each).
BASE=https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/main/data/opus/en
for split in train dev test; do
  n=$(curl -sfL "$BASE/$split/n_files.txt")
  for ((i = 0; i < n; i++)); do
    w=$(curl -sfL -r 0-300000 "$BASE/$split/audio/$i.tar.gz" | { gzip -dc 2>/dev/null || true; } |
        { tar -t 2>/dev/null || true; } | grep -m1 '\.opus$' | sed 's#.*/##; s/_common_voice.*//')
    echo "$split $i $w"
  done
done
