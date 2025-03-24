#!/bin/bash
# pip3 install tensorboard accelerate torchvision lmdb transformers diffusers tensorly pandas datasets torchmetrics torch_fidelity

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $gpu_count"

gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader)
echo "GPU IDs: $gpu_ids"

images_cnt=$1
data_name=$2
model=$3
output=$4
eval_bs=$5
resolution=$6
pred_type=$7

per_device_imges_cnt=$((images_cnt / gpu_count))

start_no=0
for gpu_id in $gpu_ids; do
    echo "Running on GPU $gpu_id"
    echo "python3 eval.py  --model_config_name_or_path="$model" --prediction_type="$pred_type" --generate_images_cnt="$per_device_imges_cnt"  --dataset_name="$data_name"   --resolution="$resolution" --output_dir="$output" --eval_batch_size="$eval_bs" --start_no="$start_no" "
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 eval.py  --model_config_name_or_path="$model" --prediction_type="$pred_type" --generate_images_cnt="$per_device_imges_cnt"  --dataset_name="$data_name"  --resolution="$resolution" --output_dir="$output" --eval_batch_size="$eval_bs" --start_no="$start_no" &
    start_no=$((start_no + per_device_imges_cnt))
done

wait

python3 eval.py --model_config_name_or_path $model --cal_metrics  --dataset_name $data_name --resolution=$resolution --output_dir $output --eval_batch_size 256