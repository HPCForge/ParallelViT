#!/bin/bash

#SBATCH -p gpu
#SBATCH -A CLASS-EECS224-GPU
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=196G
#SBATCH --gres=gpu:A30:2
#SBATCH -t 2:00:00
#SBATCH --job-name=vit


module load cuda/11.7.1
source /data/homezvol3/srachaba/envs/sciml/bin/activate

#nvcc --version

#nvcc test.cu -o test
#time ./test

#python3 -u inference_parallelformers.py
python3 -u src/parallelize.py