#!/bin/bash
#SBATCH --job-name=PCNinit
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:a100:1
#SBATCH --account=PAS0350
#SBATCH --output=/users/PAS0350/rubaolee2008/Jack_rose1775/Sphota/logs/PCNinit.log

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

# Small init (--pcn-init-std): Deepity's default init (sd ~0.08) lets settling pull each ReLU unit's
# mean incoming weight negative until the unit dies. Locally at 30k tokens, sd 0.02 took DKP
# (linear head) from 13% to 35% alone; the full PCN at 3 steps was 40% (default init).
# Reference at 300k (full PCN, 2 layers, 3 steps, default init): 46.50% alone, 55.59% / 2.219 bits/char.
D="--hidden-layers 2 --pcn-class dkp --psi-optimizer ADAMW --pcn-lmbda 1e-5 --pcn-init-std 0.02"

# 1. DKP with small init: settling steps, and whether the direct feedback matters (fb0).
run h_dkp_t1       $D --pcn-train-steps 1
run h_dkp_t2       $D --pcn-train-steps 2
run h_dkp_t3       $D --pcn-train-steps 3
run h_dkp_t2_fb0   $D --pcn-train-steps 2 --pcn-feedback-rate 0
run h_dkp_t1_fb0   $D --pcn-train-steps 1 --pcn-feedback-rate 0
run h_dkp_t2_s03   $D --pcn-train-steps 2 --pcn-init-std 0.03

# 2. Full PCN with small init: does it fix the 4-step divergence, and does it help at 3 steps?
run h_full_t3      --hidden-layers 2 --pcn-train-steps 3 --pcn-init-std 0.02
run h_full_t4      --hidden-layers 2 --pcn-train-steps 4 --pcn-init-std 0.02

# 3. Stacking on the full PCN with small init (it hurt at the default init).
run h_full_t3_stack --hidden-layers 2 --pcn-train-steps 3 --pcn-init-std 0.02 --stack-memory

# 4. Backprop control on DKP's objective (squared error instead of cross-entropy), to split the
#    gap to the cross-entropy MLP (49.07% alone at 300k) into objective cost and learning-rule cost.
run h_mlp_mse      --hidden-layers 2 --model-type mlp --mlp-loss mse --label-smooth 0.1

wait
