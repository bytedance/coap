## Training an unconditional diffusion model

Install experiment dependencies

```bash
pip install -r requirements.txt
```

Initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config default
```

The command to train a DDPM UNet model on the CIFAR-10 dataset:

```bash
# COAP-Adam, 8xV100, 1 Node
accelerate launch  --multi_gpu train_unconditional.py \
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
   --optimizer "coap_adamw" \
   --dropout 0.1 \
   --update_interval 128 \
   --reproject_factor 10 \
   --rank_ratio_matrix 1.5 \
   --rank_ratio_cnn 1.5 
```

```bash
# Generating images and calculating FID
sh eval.sh 50000 uoft-cs/cifar10 output_cifar_COAP1.5x/unet cifar10_samples 2500 32 epsilon
```

The command to train a DDPM UNet model on the CelebA-HQ dataset:

```bash
# COAP-Adam, 8xV100, 1 Node
accelerate launch  train_unconditional.py \
  --model_config_name_or_path "google/ddpm-celebahq-256" \
  --dataset_name "jxie/celeba-hq" \
  --cache_dir "data/celebahq" \
  --resolution 256 \
  --center_crop \
  --random_flip \
  --output_dir "output_celeba_COAP1.5x" \
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
  --optimizer "coap_adamw" \
  --dropout 0.1 \
  --update_interval 32 \
  --reproject_factor 10 \
  --rank_ratio_matrix 1.5 \
  --rank_ratio_cnn 1.5 
```

```bash
# Generating images and calculating FID
sh eval.sh 50000 jxie/celeba-hq output_celeba_COAP2x/unet celeba_samples 64 256 epsilon
```