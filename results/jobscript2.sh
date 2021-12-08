#!/bin/bash
#SBATCH -A e31408               # Allocation
#SBATCH -p gengpu                # Queue
#SBATCH --gres=gpu:a100:1
#SBATCH -t 06:00:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=24G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=6     # Number of Cores (Processors)
#SBATCH --mail-user=grahamgrieb2023@u.northwestern.edu  # Designate email address for job communications
#SBATCH --mail-type=END    # Events options are job BEGIN, END, NONE, FAIL, REQUEUE
#SBATCH --output=output2    # Path for output must already exist
#SBATCH --job-name="test2"       # Name of job
source activate /projects/e31408/users/gmg0603/project/env
python create_predictions_with_topn_retrieval.py