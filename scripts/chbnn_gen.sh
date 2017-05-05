#!/bin/sh

#SBATCH --job-name=chbnn_gen.py
#SBATCH --output=chb05thurs2.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
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
echo 'CHB05 simple (thresh=0.8, osr=4, usp=0.8)'
#          subj num_epochs thresh osr usp tiger tag
python ./chbnn_gen.py chb05 50 0.8 4 0.8 True thurs2
