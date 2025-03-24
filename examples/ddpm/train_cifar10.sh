# COAP-Adam, 8xV100, 1 Node

accelerate launch --multi_gpu train_unconditional.py \
  --model_config_name_or_path "google/ddpm-cifar10-32" \
  --dataset_name "uoft-cs/cifar10" \
  --cache_dir "data/cifar10" \
  --resolution 32 \
  --center_crop \
  --output_dir "output_cifar_COAP1.5x" \
  --train_batch_size 16 \
  --num_epochs 2048 \
  --gradient_accumulation_steps 1 \
  --use_ema \
  --learning_rate 2e-4 \
  --lr_warmup_steps 5000 \
  --mixed_precision no \
  --adam_weight_decay 0 \
  --random_flip \
  --eval_batch_size 128 \
  --lr_scheduler "cosine" \
  --dropout 0.1 \
  --optimizer "coap_adamw" \
  --update_interval 128 \
  --reproject_factor 10 \
  --rank_ratio_matrix 1.5 \
  --rank_ratio_cnn 1.5 