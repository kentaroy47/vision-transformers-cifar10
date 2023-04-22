#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=3:20:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

module load gcc/8.4.0-cuda python/3.7.7 py-torch/1.6.0-cuda-openmp  py-torchvision/0.6.1
python3 train_cifar10.py --q_eps 0.442 --k_eps 0.625