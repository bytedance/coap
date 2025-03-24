# LLaMA-1B, COAP-AdamW, 8 H100, 1 Node

export WANDB_MODE=offline

torchrun --standalone --nproc_per_node 8 torchrun_main.py \
    --model_config llama_configs/llama_1b.json \
    --lr 0.01 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 10000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 100000 \
    --save_every 100000 \
    --optimizer coap_adamw \
    --coap_scale 0.25 \
    --coap_rank 512 \
    --update_interval 40 \
    --reproject_factor 5 \
    --c4_data_dir "c4_data/datasets--allenai--c4"