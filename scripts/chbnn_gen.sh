#!/bin/sh

#SBATCH --job-name=chbnn_gen.py
#SBATCH --output=chb01_testsmall.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --workdir=/home/acellon/thesis
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --mail-type=end
#SBATCH --mail-user=acellon@princeton.edu

# can remove following line if it is in your ~/.bash_profile etc
export MODULEPATH=/tigress/PNI/modulefiles:$MODULEPATH

module load anaconda
module load cudatoolkit
module load intel-mkl
source activate /tigress/acellon/theano
echo 'CHB01 simple (thresh=0.8, learning_rate=1e-4)'
python ./chbnn_gen.py chb01 2 0.8 True testsmall
