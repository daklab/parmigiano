#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH --mem=15G
#SBATCH -t 0-10:00 # Runtime in D-HH:MM
#SBATCH -J skat # <-- name of job
#SBATCH --array=0-21 # <-- number of jobs to run 
#SBATCH --output=bash_outputs/tstdout_%j.log               # Standard output and error log
#SBATCH --error=bash_outputs/terror_%j.log
#SBATCH --mail-user=adas@nygenome.org 


#load required modules
module load cuda/10.0
source /gpfs/commons/home/adas/miniconda3/bin/activate
conda activate pyro

params=("inputs.yaml")
cell_index=$((SLURM_ARRAY_TASK_ID % ${#params[@]}))
params=${params[$cell_index]}
chr=$((SLURM_ARRAY_TASK_ID / ${#params[@]} + 1))
echo $chr
echo $params

python per_gene.py $params $chr
