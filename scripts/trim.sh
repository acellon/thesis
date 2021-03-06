#!/bin/sh

#SBATCH --job-name=dataTrim.py
#SBATCH --output=dataTrim.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --workdir=/tigress/acellon/data
#SBATCH --mem=32G
#SBATCH --mail-type=end
#SBATCH --mail-user=acellon@princeton.edu

# can remove following line if it is in your ~/.bash_profile etc
export MODULEPATH=/tigress/PNI/modulefiles:$MODULEPATH

module load anaconda
python /home/acellon/thesis/dataTrim.py chb16
python /home/acellon/thesis/dataTrim.py chb17
python /home/acellon/thesis/dataTrim.py chb18
python /home/acellon/thesis/dataTrim.py chb19

