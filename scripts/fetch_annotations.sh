#!/usr/bin/env bash
# Fetches the HOI-Diff annotation files (not included in this repository) from the official
# HOI-Diff repo: text descriptions, train/test split, object down-sampling indices, BPS basis.
#   https://github.com/neu-vi/HOI-Diff  (dataset/)
set -e
tmp=$(mktemp -d)
git clone --depth 1 --filter=blob:none --sparse https://github.com/neu-vi/HOI-Diff.git "$tmp/HOI-Diff"
git -C "$tmp/HOI-Diff" sparse-checkout set dataset
mkdir -p dataset
cp -r "$tmp/HOI-Diff/dataset/." dataset/
rm -rf "$tmp"
echo "annotations: $(ls dataset/behave_t2m/texts | wc -l) BEHAVE text files"
