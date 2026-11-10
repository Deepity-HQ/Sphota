#!/bin/bash
#SBATCH --job-name=PCNdkp
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:a100:1
#SBATCH --account=PAS0350
#SBATCH --output=/users/PAS0350/rubaolee2008/Jack_rose1775/Sphota/logs/PCNdkp.log

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

# Reference from the last job (2 layers, 3 steps, no stacking): 46.50% alone, 55.59% / 2.219 bits/char.

# 1. Stacking: the network also reads the n-gram's distribution, so it learns corrections to it.
run g_stack_t3     --hidden-layers 2 --pcn-train-steps 3 --stack-memory
run g_stack_t3_fl0 --hidden-layers 2 --pcn-train-steps 3 --stack-memory --pcn-feedback-rate 0

# 2. Settling steps: does 2 layers keep improving past 3 steps, and what do 1 layer need?
run g_L2_t4        --hidden-layers 2 --pcn-train-steps 4
run g_L1_t3        --hidden-layers 1 --pcn-train-steps 3
run g_L1_t4        --hidden-layers 1 --pcn-train-steps 4

# 3. Direct Kolen-Pollack (one-step error to every layer, squared-error head). Local CPU tests
#    were weak (<16% at 30k tokens) and diverged at feedback 1e-3, so keep the rates small.
run g_dkp_t1       --hidden-layers 2 --pcn-class dkp --pcn-train-steps 1 --pcn-lmbda 1e-5
run g_dkp_t2       --hidden-layers 2 --pcn-class dkp --pcn-train-steps 2 --pcn-lmbda 1e-5
run g_dkp_t2_fb    --hidden-layers 2 --pcn-class dkp --pcn-train-steps 2 --pcn-lmbda 1e-5 --pcn-feedback-rate 1e-5

wait
