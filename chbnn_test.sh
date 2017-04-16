#!/bin/sh

#SBATCH --job-name=chbnn.py
#SBATCH --output=chbnn.o%j
#SBATCH --error=chbnn.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --workdir=/home/acellon/thesis
#SBATCH --gres=gpu:2
#SBATCH --mem=7G
#SBATCH --mail-type=end
#SBATCH --mail-user=acellon@princeton.edu

# can remove following line if it is in your ~/.bash_profile etc
export MODULEPATH=/tigress/PNI/modulefiles:$MODULEPATH

module load anaconda
module load cudatoolkit
module load intel-mkl
source activate /tigress/acellon/theano
python ./chbnn.py chb01 100 True

