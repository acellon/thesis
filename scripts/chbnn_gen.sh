#!/bin/sh

#SBATCH --job-name=chb24init
#SBATCH --output=chb24init.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
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
echo 'CHB24 simple, thresh=0.8, osr=4, usp=0.6, learn=1e-5'
#          subj num_epochs thresh osr usp tiger tag
python ./chbnn_gen.py chb24 50 0.8 4 0.6 True init
