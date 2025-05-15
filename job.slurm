#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=<haralddj>
#SBATCH --time=0-00:15:00     # 0 days and 15 minutes limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=2     # 2 CPU cores
#SBATCH --mem=5G              # 5 gigabytes memory
#SBATCH --output=hello.txt    # Log file
echo "Hello IDUN"
