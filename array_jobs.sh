#!/bin/bash
#SBATCH --partition=3090-gcondo         # Partition to run on
#SBATCH --gres=gpu:1             # Request 1 GPU resources
#SBATCH --time=2-00:00:00          # Request 12 hours of runtime
#SBATCH --mem=32G                # Request 32GB of memory
#SBATCH -J mae_linprobes        # Specify a job name
#SBATCH -o mae_linprobes-%j.out # Specify an output file
#SBATCH -e mae_linprobes-%j.err # Specify an error file
#SBATCH --array=1-18

source venv/bin/activate

TO_RUN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" linprobe_runs.txt)
eval $TO_RUN
