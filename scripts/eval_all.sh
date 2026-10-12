#!/usr/bin/env bash
# Samples the BEHAVE test set with every trained model, then computes all metrics.
set -e
mkdir -p logs results
J() { local name=$1; shift; sbatch --parsable -J "$name" scripts/slurm_job.sh "$@"; }
GEN="python -m sample.generate_eval --guidance --batch_size 64 --seed 10"
declare -A APDM=( [hoidiff]=apdm_base [rethoi]=apdm_con [wo_retnet]=apdm_con [wo_gabor]=apdm_con [wo_con]=apdm_base )
ids=()
for m in hoidiff rethoi wo_retnet wo_gabor wo_con; do
  ids+=($(J gen_$m $GEN --model_path save/$m/model000020000.pt \
          --afford_model_path save/${APDM[$m]}/model000010000.pt --out results/$m.npz))
done
dep=$(IFS=:; echo "${ids[*]}")
sbatch --parsable -J metrics --dependency=afterok:$dep scripts/slurm_job.sh python -m eval.eval_behave \
   --samples results/hoidiff.npz results/rethoi.npz results/wo_retnet.npz results/wo_gabor.npz results/wo_con.npz \
   --names HOI-Diff RetHOI "w/o RetNet" "w/o Gabor" "w/o Contrastive" --out results/metrics.json
