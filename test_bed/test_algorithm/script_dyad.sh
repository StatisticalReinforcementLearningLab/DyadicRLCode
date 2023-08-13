#!/bin/bash

# Now run normal batch commands
seed=$1
one=$2
two=$3
three=$4

module load Anaconda3/2020.11
source activate myenv
python3 dyad_cluster.py $seed $one $two $three
