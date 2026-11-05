#!/bin/bash
#SBATCH --job-name=PCNdiag
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:a100:1
#SBATCH --account=PAS0350
#SBATCH --output=/users/PAS0350/rubaolee2008/Jack_rose1775/Sphota/logs/PCNdiag.log

module load cuda/13.2.1
module load cudnn/9.21.1.3-13
source .venv/bin/activate
export PYTHONUNBUFFERED=1

L=/users/PAS0350/rubaolee2008/Jack_rose1775/Sphota/logs
B="--tokens 300000 --window 4 --hidden-layers 2 --embed-lr 1e-4 --log-every 100000"

# run <log name> <extra args...>: one background run, output in $L/<name>.log
run() {
  name=$1; shift
  uv run python -m sphota.main $B "$@" > "$L/$name.log" 2>&1 &
}

# Feedback on, with the bounding decay (Deepity's W *= 1-lmbda per step).
run d_fl_lmb1e-5 --model-type pcn --lr 1e-4 --pcn-feedback-rate 1e-4 --pcn-lmbda 1e-5
run d_fl_lmb1e-4 --model-type pcn --lr 1e-4 --pcn-feedback-rate 1e-4 --pcn-lmbda 1e-4

# Feedback off: lower rate, decayed rate, frozen embeddings.
run d_lr3e-5     --model-type pcn --lr 3e-5 --pcn-feedback-rate 0
run d_decay      --model-type pcn --lr 3.3e-4 --lr-halflife 50000 --pcn-feedback-rate 0
run d_frozen     --model-type pcn --lr 1e-4 --pcn-feedback-rate 0 --freeze-embeddings

# Control: same network with exact gradients, frozen embeddings.
run d_mlp_frozen --model-type mlp --lr 1e-4 --freeze-embeddings

wait
