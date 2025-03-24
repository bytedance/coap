# COAP-Adam, 8xV100, 1 Node

accelerate launch --multi_gpu train_unconditional.py \
  --model_config_name_or_path "google/ddpm-celebahq-256" \
  --dataset_name "jxie/celeba-hq" \
  --cache_dir "data/celebahq" \
  --resolution 256 \
  --center_crop \
  --random_flip \
  --output_dir "output_celeba_COAP2x" \
  --train_batch_size 8 \
  --num_epochs 1067 \
  --gradient_accumulation_steps 1 \
  --use_ema \
  --learning_rate 2e-5 \
  --lr_warmup_steps 5000 \
  --mixed_precision no \
  --use_ema \
  --adam_weight_decay 0 \
  --random_flip \
  --eval_batch_size 8 \
  --dropout 0.1 \
  --optimizer "coap_adamw" \
  --update_interval 32 \
  --reproject_factor 10 \
  --rank_ratio_matrix 2 \
  --rank_ratio_cnn 2