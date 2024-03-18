PORT=$(( $RANDOM % 1000 + 32768 ))
# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# force crashing on nccl issues like hanging broadcast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GRUB_CMDLINE_LINUX_DEFAULT="iommu=soft"
export CUDA_VISIBLE_DEVICES=$1
export NUM_PROCESSES=$2
export DATA_DIR=/mnt/users/n3thakur/vectara/vectara-translation/datasets/mistral-translate/mistralai/Mistral-7B-Instruct-v0.2/argilla/postprocessed/

# Mistral-ORPO series are trained on 4 * A100s
accelerate launch --main_process_port $PORT --num_processes $NUM_PROCESSES --config_file ./configs/fsdp.yaml ../main.py \
    --lr 5e-6 \
    --lr_scheduler_type inverse_sqrt \
    --alpha 0.1 \
    --torch_compile False \
    --warmup_steps 200 \
    --model_name mistralai/Mistral-7B-v0.1 \
    --data_name $DATA_DIR/distilabel-intel-orca-dpo-pairs-processed-v2.jsonl \
    --cache_dir /mnt/users/n3thakur/cache \
    --num_train_epochs 1 \
    --prompt_max_length 1792 \
    --response_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_proc 1 \
    --flash_attention_2 

accelerate launch --main_process_port $PORT --num_processes $NUM_PROCESSES --config_file ./configs/fsdp.yaml ../main.py \
    --lr 5e-6 \
    --lr_scheduler_type inverse_sqrt \
    --alpha 0.1 \
    --torch_compile False \
    --warmup_steps 200 \
    --model_name google/gemma-2b \
    --data_name $DATA_DIR/distilabel-intel-orca-dpo-pairs-processed-v2.jsonl \
    --cache_dir /mnt/users/n3thakur/cache \
    --num_train_epochs 1 \
    --prompt_max_length 1792 \
    --response_max_length 2048 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_proc 1 \
    --flash_attention_2
