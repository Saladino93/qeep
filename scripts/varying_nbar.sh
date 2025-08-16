#!/bin/bash
#SBATCH --account=lp44
#SBATCH --job-name=varying_nbar-%j
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=normal

#uenv start --view=modules b50ca0d101456970
module load python cray-mpich nccl cuda fftw gcc hdf5 openblas cmake fmt meson libtree
source /users/odarwish/lenscarf/bin/activate

echo "Starting job on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS tasks"
echo "Job ID: $SLURM_JOB_ID"

# Run the simple Python script with srun
srun python varying_nbar.py --force

echo "Job completed"







