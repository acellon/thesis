#!/bin/sh

#SBATCH --job-name=chb05samp15
#SBATCH --output=chb05samp15.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --workdir=/home/acellon/thesis
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --mail-type=end
#SBATCH --mail-user=acellon@princeton.edu

# can remove following line if it is in your ~/.bash_profile etc
export MODULEPATH=/tigress/PNI/modulefiles:$MODULEPATH

module load anaconda
module load cudatoolkit
module load intel-mkl
source activate /tigress/acellon/theano
echo 'CHB05 simple, thresh=0.8, osr=5, usp=0.86899052, learn=1e-5'
#          subj num_epochs thresh osr usp tiger tag
python ./chbnn_gen.py chb05 50 0.8 5 0.86899052 True samp15
