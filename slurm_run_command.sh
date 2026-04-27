#!/bin/bash
#SBATCH ... with gpu options

echo "Running on gpu: $@"

# module setup
# ...

srun "$@"