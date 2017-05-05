#!/bin/sh

#SBATCH --job-name=chbnn_gen.py
#SBATCH --output=chb09testlw.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
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
echo 'CHB09 simple test of loowin (thresh=0.75, learning_rate=1e-4)'
python ./chbnn_gen.py chb09 2 0.75 True testlw
