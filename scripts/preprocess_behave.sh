#!/usr/bin/env bash
# BEHAVE -> HumanML3D-style HOI representation (see process/README.md).
# Expects ./dataset/raw_data/{behave_raw,behave_objects} and ./body_models/smplh/SMPLH_{MALE,FEMALE}.npz
# The per-sequence steps run in NUM_SHARDS parallel processes.
set -e
export NUM_SHARDS=${NUM_SHARDS:-12} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

sharded() {  # runs "$@" once per shard and fails if any shard fails
  local pids=()
  for ((i = 0; i < NUM_SHARDS; i++)); do SHARD_ID=$i "$@" > "logs/preprocess_shard$i.log" 2>&1 & pids+=($!); done
  for pid in "${pids[@]}"; do wait "$pid"; done
}
mkdir -p logs
ONLY_OBJECTS=1 python process/process_behave.py
SKIP_OBJECTS=1 sharded python process/process_behave.py
sharded python process/motion_representation_263.py --dataset behave
python process/cal_mean_variance.py --dataset behave
AFFORD_DEVICE=cpu sharded python process/get_affordance.py --dataset behave  # 12 SMPL-H models do not fit on one GPU
echo "sequences: $(ls dataset/behave_t2m/sequences_263_rep | wc -l)  affordance: $(ls dataset/behave_t2m/affordance | wc -l)"
