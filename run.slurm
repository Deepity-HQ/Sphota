#!/bin/bash
#SBATCH --job-name=FeasibilityTest
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:a100:1
#SBATCH --account=PAS0350
#SBATCH --output=/users/PAS0350/rubaolee2008/Jack_rose1775/Sphota/logs/FeasibilityTest.log

module load cuda/13.2.1
module load cudnn/9.21.1.3-13

source .venv/bin/activate

uv run python -m sphota.main