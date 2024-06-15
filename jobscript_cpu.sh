#!/bin/bash

#SBATCH -p standard
#SBATCH -A amowli_lab
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=196G
#SBATCH -t 12:00:00
#SBATCH --job-name=vit


source /data/homezvol3/srachaba/envs/sciml/bin/activate

#python3 -u plot.py
#python3 -u plot.py
python3 -u src/parallelize.py
