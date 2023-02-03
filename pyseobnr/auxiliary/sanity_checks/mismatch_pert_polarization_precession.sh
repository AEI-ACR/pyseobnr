#!/bin/bash -
#SBATCH -J v5PHM_pert_test
#SBATCH -o v5PHM_pert_test.stdout           # Output file name
#SBATCH -e v5PHM_pert_test.stderr           # Error file name
#SBATCH -n 64                # Number of cores
#SBATCH --ntasks-per-node 64         # number of MPI ranks per node
#SBATCH -p syntrofos                 # Queue name
#SBATCH -t 72:00:00            # Run time
#SBATCH --no-requeue

source /home/aramosbuades/load_LALpyseobnr.sh

export HDF5_USE_FILE_LOCKING='FALSE'
export OMP_NUM_THREADS=1

python /work/aramosbuades/git/pyseobnr_update_precessing_v2/pyseobnr/auxiliary/sanity_checks/mismatch_pert_polarization_precession.py --points 100000 --M-min 10 --M-max 300 --q-max 100 --plots --ncores 64 --ell-max 4 --omega0 0.018
