#!/usr/bin/env bash
# Submits every training run of the paper (main model, baseline and ablations) to Slurm.
#   APDM:   apdm_base (HOI-Diff)            apdm_con (+ InfoNCE)
#   HOI-DM: hoidiff   (Transformer)         rethoi   (RetNet + Gabor + InfoNCE)
#           wo_retnet / wo_gabor / wo_con   (one component removed each)
set -e
mkdir -p logs save
J() { local name=$1; shift; sbatch --parsable -J "$name" scripts/slurm_job.sh "$@"; }
APDM="python -m train.train_affordance --dataset behave --batch_size 32 --diffusion_steps 500 --num_steps 10000 --save_interval 2500 --overwrite"
HOIDM="python -m train.train_hoi_diff --dataset behave --batch_size 32 --num_steps 20000 --save_interval 5000 --multi_backbone_split 4 --overwrite"
OURS="--lambda_smooth 0.05 --lambda_con 0.1"

J apdm_base  $APDM --save_dir save/apdm_base
J apdm_con   $APDM --save_dir save/apdm_con --use_contrastive --lambda_con 0.1
J hoidiff    $HOIDM --save_dir save/hoidiff   --arch trans_enc
J rethoi     $HOIDM --save_dir save/rethoi    --arch retnet    --use_gabor --use_contrastive $OURS
J wo_retnet  $HOIDM --save_dir save/wo_retnet --arch trans_enc --use_gabor --use_contrastive $OURS
J wo_gabor   $HOIDM --save_dir save/wo_gabor  --arch retnet    --use_contrastive $OURS
J wo_con     $HOIDM --save_dir save/wo_con    --arch retnet    --use_gabor $OURS
J evaluator  python -m eval.evaluator --save_path checkpoints/behave_evaluator.pt
squeue -u "$USER"
