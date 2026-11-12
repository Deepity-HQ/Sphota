#!/bin/bash
#SBATCH --job-name=DKPseeds
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:a100:1
#SBATCH --account=PAS0350
#SBATCH --output=/users/PAS0350/rubaolee2008/Jack_rose1775/Sphota/logs/DKPseeds.log

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

# Does DKP's direct feedback help? One hidden layer, linear head, 1 settling step, no decay.
# At 100k tokens locally (one seed), two settings beat feedback-off:
#   SGD on W and Psi, random Psi:   36.24% vs 31.66% off  (cos(Psi, W^T) grew to +0.35)
#   ADAMW on both, aligned Psi:     31.84% vs 27.66% off  (cos stays +1.000)
# Here: 300k tokens, 3 seeds, each setting against feedback-off. Each log prints cos(Psi, W^T).
D="--hidden-layers 1 --pcn-class dkp --pcn-train-steps 1 --pcn-lmbda 0"
SGD="--pcn-optimizer SGD --psi-optimizer SGD --lr 1e-3"
ADAM="--pcn-optimizer ADAMW --psi-optimizer ADAMW --lr 1e-4"

for s in 0 1 2; do
  run k_sgd_on_s$s    $D $SGD  --seed $s --pcn-feedback-rate 1e-3
  run k_sgd_off_s$s   $D $SGD  --seed $s --pcn-feedback-rate 0
  run k_adam_al_s$s   $D $ADAM --seed $s --pcn-feedback-rate 1e-4 --dkp-align-psi
  run k_adam_off_s$s  $D $ADAM --seed $s --pcn-feedback-rate 0
done

wait
