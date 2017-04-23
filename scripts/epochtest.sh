#!/bin/sh

#SBATCH --job-name=epochtest.py
#SBATCH --output=epochtest.o%j
#SBATCH --error=epochtest.e%j
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --workdir=/home/acellon/thesis
#SBATCH --mem=60G
#SBATCH --mail-type=end
#SBATCH --mail-user=acellon@princeton.edu

# can remove following line if it is in your ~/.bash_profile etc
export MODULEPATH=/tigress/PNI/modulefiles:$MODULEPATH

module load anaconda
module load cudatoolkit
module load intel-mkl
source activate /tigress/acellon/theano
python ./epochtest.py chb16

