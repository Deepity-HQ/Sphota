#!/bin/bash
#SBATCH --job-name=PCNsteps
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:a100:1
#SBATCH --account=PAS0350
#SBATCH --output=/users/PAS0350/rubaolee2008/Jack_rose1775/Sphota/logs/PCNsteps.log

module load cuda/13.2.1
module load cudnn/9.21.1.3-13
source .venv/bin/activate
export PYTHONUNBUFFERED=1

L=/users/PAS0350/rubaolee2008/Jack_rose1775/Sphota/logs
B="--tokens 300000 --window 4 --lr 1e-4 --embed-lr 1e-4 --model-type pcn --log-every 100000"

# run <log name> <extra args...>: one background run, output in $L/<name>.log
run() {
  name=$1; shift
  uv run python -m sphota.main $B "$@" > "$L/$name.log" 2>&1 &
}

# A network with H hidden layers needs H+1 settling steps before its first layer sees any error.
# 2 hidden layers: 1 step (old default), 2 steps, 3 steps (should be enough).
run s_L2_t1 --hidden-layers 2 --pcn-train-steps 1
run s_L2_t2 --hidden-layers 2 --pcn-train-steps 2
run s_L2_t3 --hidden-layers 2 --pcn-train-steps 3
# 1 hidden layer: 1 step (old), 2 steps (should be enough).
run s_L1_t1 --hidden-layers 1 --pcn-train-steps 1
run s_L1_t2 --hidden-layers 1 --pcn-train-steps 2
# Enough steps, direct feedback off: is the learning rule itself now carrying the hidden layers?
run s_L2_t3_fl0 --hidden-layers 2 --pcn-train-steps 3 --pcn-feedback-rate 0

wait
