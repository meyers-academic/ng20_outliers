#!/bin/bash
#SBATCH --job-name=outlier_run
#SBATCH --account=a157
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --array=0-0

# read in psr_names.txt under datasets directory
readarray -t pulsars < ../datasets/psr_names.txt

PULSAR_NAME=${pulsars[$SLURM_ARRAY_TASK_ID]}

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Pulsar: $PULSAR_NAME"

/users/atresnjic/miniforge3/envs/discovery/bin/python -c "import discovery"
/users/atresnjic/miniforge3/envs/discovery/bin/python /users/atresnjic/projects/test_run/scripts/run_outlier.py -p "$PULSAR_NAME" -w 100 -N 1000 -n 30 -o /capstor/scratch/cscs/atresnjic/outlier_results/ -r True -c 200
~
~
~
~