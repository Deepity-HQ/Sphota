#!/bin/bash
#SBATCH --job-name=DKPinit
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:a100:1
#SBATCH --account=PAS0350
#SBATCH --output=/users/PAS0350/rubaolee2008/Jack_rose1775/Sphota/logs/DKPinit.log

module load cuda/13.2.1
module load cudnn/9.21.1.3-13
source .venv/bin/activate
export PYTHONUNBUFFERED=1

L=/users/PAS0350/rubaolee2008/Jack_rose1775/Sphota/logs
B="--tokens 300000 --window 4 --lr 1e-4 --embed-lr 1e-4 --model-type pcn --log-every 50000"

# run <log name> <extra args...>: one background run, output in $L/<name>.log
run() {
  name=$1; shift
  uv run python -m sphota.main $B "$@" > "$L/$name.log" 2>&1 &
}

# Small forward init for DKP. One hidden layer, linear head, 1 settling step, no decay, 3 seeds.
# Reference at 300k (default init, SGD on W and Psi, 3 seeds): 43.48% feedback on, 34.46% off.
# Questions: does sd 0.01 lift SGD above 43.5%, and does ADAMW hold at 300k or run away?
# Each log prints cos(Psi, W^T), |Psi| and |W| every 50k tokens.
D="--hidden-layers 1 --pcn-class dkp --pcn-train-steps 1 --pcn-lmbda 0 --pcn-init-std 0.01"
SGD="--pcn-optimizer SGD --psi-optimizer SGD --lr 1e-3"
ADAM="--pcn-optimizer ADAMW --psi-optimizer ADAMW --lr 1e-4"

for s in 0 1 2; do
  run a_sgd_on_s$s    $D $SGD  --seed $s --pcn-feedback-rate 1e-3
  run b_sgd_off_s$s   $D $SGD  --seed $s --pcn-feedback-rate 0
  run c_adam_on_s$s   $D $ADAM --seed $s --pcn-feedback-rate 1e-4
  run d_adam_off_s$s  $D $ADAM --seed $s --pcn-feedback-rate 0
done

wait
