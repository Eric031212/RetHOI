#!/usr/bin/env bash
# Downloads the public dependencies (GloVe, SMPL for visualisation/rot2xyz, pretrained MDM).
set -e
mkdir -p body_models checkpoints
[ -d glove ] || { gdown 1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n -O glove.zip && unzip -q glove.zip && rm glove.zip; }
[ -d body_models/smpl ] || { (cd body_models && gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2" && unzip -q smpl.zip && rm smpl.zip); }
[ -d checkpoints/humanml_trans_enc_512 ] || { (cd checkpoints && gdown 1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821 && unzip -q humanml_trans_enc_512.zip && rm humanml_trans_enc_512.zip); }
ls glove body_models/smpl checkpoints/humanml_trans_enc_512
