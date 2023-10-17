#!/bin/bash
#SBATCH --partition=gpu                    # Partition [compute|memory|gpu]
#SBATCH --nodes=4                          # Number of nodes
#SBATCH --ntasks-per-node=1                # Tasks per node
#SBATCH --cpus-per-task=64                 # CPUs per task
#SBATCH --gpus-per-task=4                  # GPUs per task
#SBATCH --time=5-00:00:00                  # Time limit (day-hour:minute:second)
#SBATCH --account=lt200063                 # Project name
#SBATCH --job-name=train_multi_node        # Job name
#SBATCH --output=R-%x.out                  # Output file
#SBATCH --error=R-%x.out                   # Error file

ml Miniconda3
conda activate /project/lt200063-idcd/envs

export NCCL_SOCKET_IFNAME="bond0"
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TORCHELASTIC_ERROR_FILE="pytorch_error.json"

echo "NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_NAMES:"

srun bash job/local_train_multi_node.sh
