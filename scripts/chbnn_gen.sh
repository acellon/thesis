#!/bin/sh

#SBATCH --job-name=chb21samp9
#SBATCH --output=chb21samp9.o%j
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
echo 'CHB21 simple, thresh=0.8, osr=6, usp=0.28549516, learn=1e-5'
#          subj num_epochs thresh osr usp tiger tag
python ./chbnn_gen.py chb21 50 0.8 6 0.28549516 True samp9
