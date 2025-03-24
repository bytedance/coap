# LLaMA-7B, COAP-AdamW8bit, 8 H100, 1 Node

export WANDB_MODE=offline

torchrun --standalone --nproc_per_node 8 torchrun_main.py \
    --model_config llama_configs/llama_7b.json \
    --lr 0.005 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 150000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 150000 \
    --save_every 150000 \
    --activation_checkpointing \
    --optimizer coap_adamw8bit \
    --coap_scale 0.25 \
    --coap_rank 1024 \
    --update_interval 100 \
    --reproject_factor 5 \
    --c4_data_dir "c4_data/datasets--allenai--c4"