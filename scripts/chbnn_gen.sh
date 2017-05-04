#!/bin/sh

#SBATCH --job-name=chbnn_gen.py
#SBATCH --output=chb09_winbig.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --workdir=/home/acellon/thesis
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --mail-type=end
#SBATCH --mail-user=acellon@princeton.edu

# can remove following line if it is in your ~/.bash_profile etc
export MODULEPATH=/tigress/PNI/modulefiles:$MODULEPATH

module load anaconda
module load cudatoolkit
module load intel-mkl
source activate /tigress/acellon/theano
echo 'CHB09 simple (thresh=0.9)'
python ./chbnn_gen.py chb09 25 0.9 True
