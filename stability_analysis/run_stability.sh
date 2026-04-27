#!/bin/bash

# --random_init parameter takes care of initializing ICA differently

method=$1
dim=$2
model=$3
data_config=$4
lang=$5   # this is the split, just known as lang here because it always is language


data_name=$(basename $data_config .json)    # tatoeba, etc.
model_name=$(basename $model) # remove path if given as a path
moment=$(echo $(date '+%d-%m--%H-%M-%S'))   # to differentiate runs
save_fitted="stability/${method}_dim_${dim}/${data_name}/${lang}/${model_name}_fitted_${moment}.pkl" 
save_results="stability/${method}_dim_${dim}/${data_name}/${lang}/${model_name}_results_${moment}.pkl"
save_stats="stability/${method}_dim_${dim}/${data_name}/${lang}/${model_name}_stats_${moment}.json"
embeddings="embeddings/${data_name}:${lang}/${model_name}_embeddings.pkl"

if [ -s "$save_results" ]; then
    echo "Result file exists and is not empty: exiting to avoid overwriting."
    exit 17
fi

CMD=(python fit_paired_data.py \
                    --method=$method \
                    --dim=$dim \
                    --config=$data_config \
                    --data_split=$lang \
                    --model_name=$model \
                    --embs=$embeddings \
                    --downsample=5000 \
                    --save_fitted=$save_fitted \
                    --result_path=$save_results \
                    --random_init \
                    --debug)


if [ -f $embeddings ]; then
    echo "Running on cpu and reading precalculated embeddings"
    sbatch --job-name="stability_${data_name}-${lang}_${model_name}" --output="logs/%x-%j.out" slurm_run_command_cpu.sh "${CMD[@]}"
else
    echo "Running on gpu"
    sbatch --job-name="stability_${data_name}-${lang}_${model_name}" --output="logs/%x-%j.out" slurm_run_command.sh "${CMD[@]}"
fi