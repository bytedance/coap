# ControlNet-XL, COAP-Adafactor, 8 A100, 1 Node.

export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="fill50k-ada2x"

accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "./condition/val_1.png" "./condition/val_2.png" \
 --validation_prompt "red circle with blue background" "dark cyan circle with brown background" \
 --validation_steps=100 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="tensorboard" \
 --seed=42 \
 --optimizer "coap_adafactor" \
 --update_interval 32 \
 --reproject_factor 10 \
 --rank_ratio_matrix 2 \
 --rank_ratio_cnn 2 