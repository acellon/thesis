#!/bin/sh

#SBATCH --job-name=chbnn_gen.py
#SBATCH --output=chb19init.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
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
echo 'CHB19 simple, thresh=0.8, osr=4, usp=0.6, using adam w learn=1e-5'
#          subj num_epochs thresh osr usp tiger tag
python ./chbnn_gen.py chb19 25 0.8 4 0.6 True init
