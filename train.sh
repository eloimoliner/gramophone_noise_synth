#!/bin/bash
#SBATCH  --time=2-23:59:59
#SBATCH --mem=10G
#SBATCH --job-name=diffwave
#SBATCH  --gres=gpu:a100:1
##SBATCH  --gres=gpu:1 --constraint=volta
#SBATCH --output=/scratch/work/%u/projects/ddpm/CRASH/experiments/%a/train_%j.out

#SBATCH --array=[23]

module load anaconda
source activate /scratch/work/molinee2/conda_envs/2022_torchot
export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

#
PATH_EXPERIMENT=experiments/trained_model
mkdir $PATH_EXPERIMENT

#python train_w_cqt.py path_experiment="$PATH_EXPERIMENT"  $iteration
python train.py model_dir="$PATH_EXPERIMENT"
