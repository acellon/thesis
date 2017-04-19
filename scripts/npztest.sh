#!/bin/sh

#SBATCH --job-name=npztest.py
#SBATCH --output=npztest.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --workdir=/home/acellon/thesis
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=end
#SBATCH --mail-user=acellon@princeton.edu

# can remove following line if it is in your ~/.bash_profile etc
export MODULEPATH=/tigress/PNI/modulefiles:$MODULEPATH

module load anaconda
module load cudatoolkit
module load intel-mkl
python ./npztest.py chb01
python ./npztest.py chb02
python ./npztest.py chb03
python ./npztest.py chb04
python ./npztest.py chb05
python ./npztest.py chb06
python ./npztest.py chb07
python ./npztest.py chb08
python ./npztest.py chb09
python ./npztest.py chb10
python ./npztest.py chb11
python ./npztest.py chb14
python ./npztest.py chb16
python ./npztest.py chb17
python ./npztest.py chb18
python ./npztest.py chb19
python ./npztest.py chb20
python ./npztest.py chb21
python ./npztest.py chb22
python ./npztest.py chb23
python ./npztest.py chb24
