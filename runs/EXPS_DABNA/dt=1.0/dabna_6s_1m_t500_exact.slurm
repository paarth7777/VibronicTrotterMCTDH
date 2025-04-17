#!/bin/bash
# Job name:
#SBATCH --job-name=mctdh_dabna_6s_1m_t500_exact
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paarth.jain@mail.utoronto.ca
module load gcc/13.2.0
# Command(s) to run:
mctdh86 -w dabna_6s_1m_t500_exact
