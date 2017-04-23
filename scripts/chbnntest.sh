#!/bin/sh

#SBATCH --job-name=chbnn.py
#SBATCH --output=chbnn.o%j
#SBATCH --error=chbnn.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00
#SBATCH --workdir=/home/acellon/thesis

#SBATCH --mem=60GB
#SBATCH --mail-type=end
#SBATCH --mail-user=acellon@princeton.edu

# can remove following line if it is in your ~/.bash_profile etc
export MODULEPATH=/tigress/PNI/modulefiles:$MODULEPATH

module load anaconda
module load cudatoolkit
module load intel-mkl
python ./chbnn.py chb01 5 True
