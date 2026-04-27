#!/bin/bash
#SBATCH ...

echo "Running on cpu: $@"

# module setup
# ...

srun "$@"