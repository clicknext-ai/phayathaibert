echo "$SLURMD_NODENAME: rank $SLURM_NODEID"

accelerate launch \
    --num_machines $SLURM_JOB_NUM_NODES \
    --num_processes $((4 * $SLURM_JOB_NUM_NODES)) \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --config_file config/accelerate/multi_node_multi_gpu.yaml \
    train.py \
    --model_dir checkpoints/first_run \
    --train_data train_eval_test/train \
    --eval_data train_eval_test/eval \
    --config_file config/training/training_config.toml
