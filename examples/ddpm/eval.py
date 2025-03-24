# Copyright The HuggingFace Inc.
#
# Copyright 2025 ByteDance
# Modifications made by the COAP authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import inspect
import logging
import math
import os
import re
import shutil
from datetime import timedelta
from pathlib import Path
import time
from datasets import load_dataset
import accelerate
import datasets
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo
from packaging import version
from torchvision import transforms
import diffusers
from diffusers import (
    DDPMPipeline,
    DDPMScheduler,
    UNet2DModel,
    DDIMPipeline,
    DDIMScheduler,
)
from diffusers.utils import (
    check_min_version,
    is_accelerate_version,
    is_tensorboard_available,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.0.dev0")

logger = get_logger(__name__, log_level="INFO")

from torch_fidelity import calculate_metrics
from torch.utils.data import Dataset
import numpy as np


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, resolution):
        self.dataset = hf_dataset
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255).byte()),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if 'img' in sample:
            image = sample["img"].convert("RGB")
        else:
            image = sample["image"].convert("RGB")
        image = self.transform(image)
        # image = np.array(image)
        # image = np.transpose(image, (2, 0, 1))
        # image = torch.tensor(image, dtype=torch.uint8)
        return image


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard ddpm configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="The number of images to generate for evaluation.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument(
        "--ema_inv_gamma",
        type=float,
        default=1.0,
        help="The inverse gamma value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_power",
        type=float,
        default=3 / 4,
        help="The power value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_max_decay",
        type=float,
        default=0.9999,
        help="The maximum decay magnitude for EMA.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo",
        action="store_true",
        help="Whether or not to create a private repository.",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    # parser.add_argument("--ddim_num_inference_steps", type=int, default=100)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument("--generate_images_cnt", type=int, default=50000)
    parser.add_argument("--start_no", type=int, default=0)
    parser.add_argument(
        "--cal_metrics",
        action="store_true",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def find_unet_folders(base_dir):
    unet_folders = []
    max_number = 0
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            match = re.match(r"checkpoint-(\d+)", dir_name)
            if match:
                checkpoint_number = int(match.group(1))
                checkpoint_unet_folder = os.path.join(root, dir_name, "unet")
                if os.path.exists(checkpoint_unet_folder):
                    unet_folders.append((checkpoint_unet_folder, checkpoint_number))
                    max_number = max(max_number, checkpoint_number)
    if os.path.exists(os.path.join(base_dir, "unet")):
        unet_folders.insert(0, (os.path.join(base_dir, "unet"), max_number + 1000))
    unet_folders.sort(key=lambda x: x[1], reverse=True)
    return unet_folders


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=7200)
    )  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    # if args.logger == "tensorboard":
    #     if not is_tensorboard_available():
    #         raise ImportError(
    #             "Make sure to install tensorboard if you want to use it for logging during training."
    #         )

    # elif args.logger == "wandb":
    #     if not is_wandb_available():
    #         raise ImportError(
    #             "Make sure to install wandb if you want to use it for logging during training."
    #         )
    #     import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            # if os.path.exists(args.output_dir):
            #     shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Initialize the model
    assert args.model_config_name_or_path is not None
    # paths = find_unet_folders(args.model_config_name_or_path)
    # print(paths)
    # for path, step in paths:
    #     print(step, path)
    if not args.cal_metrics:
        model = UNet2DModel.from_pretrained(args.model_config_name_or_path)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            args.mixed_precision = accelerator.mixed_precision
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            args.mixed_precision = accelerator.mixed_precision

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        # Initialize the scheduler
        accepts_prediction_type = "prediction_type" in set(
            inspect.signature(DDPMScheduler.__init__).parameters.keys()
        )
        if accepts_prediction_type:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.ddpm_num_steps,
                beta_schedule=args.ddpm_beta_schedule,
                prediction_type=args.prediction_type,
            )
            # noise_scheduler = DDIMScheduler(
            #     num_train_timesteps=args.ddpm_num_steps,
            #     beta_schedule=args.ddpm_beta_schedule,
            #     prediction_type=args.prediction_type,
            #     rescale_betas_zero_snr=True,
            #     clip_sample=False,
            #     set_alpha_to_one=False,
            # )
        else:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.ddpm_num_steps,
                beta_schedule=args.ddpm_beta_schedule,
                thresholding=True,
            )
            # noise_scheduler = DDIMScheduler(
            #     num_train_timesteps=args.ddpm_num_steps,
            #     beta_schedule=args.ddpm_beta_schedule,
            #     rescale_betas_zero_snr=True,
            #     clip_sample=False,
            #     set_alpha_to_one=False,
            # )

        # Prepare everything with our `accelerator`.
        model = accelerator.prepare(model)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            # run = os.path.split(__file__)[-1].split(".")[0]
            # accelerator.init_trackers(run)
            # Generate sample images for visual inspection
            unet = accelerator.unwrap_model(model)
            # pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
            pipeline = DDIMPipeline(unet=unet, scheduler=noise_scheduler)
            generator = torch.Generator(device=pipeline.device).manual_seed(
                args.start_no
            )
            # args.start_no + int(time.time())
            # generator = torch.Generator(device=pipeline.device)
            # run pipeline in inference (sample random noise and denoise)

            cnt = args.generate_images_cnt
            while cnt > 0:
                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps,
                ).images
                for img in images:
                    if cnt > 0:
                        file = os.path.join(
                            args.output_dir, f"image_{args.start_no + cnt}.png"
                        )
                        cnt -= 1
                        img.save(file)
    if accelerator.is_main_process and args.cal_metrics:
        # log = os.path.join(args.output_dir, args.logging_dir, "log.txt")
        log = args.output_dir + ".log"
        dataset = load_dataset(
            args.dataset_name,
            split="train",
        )
        wrapped_dataset = HFDatasetWrapper(dataset, args.resolution)
        metrics = calculate_metrics(
            input1=args.output_dir,
            input2=wrapped_dataset,
            cuda=True,
            isc=True,
            fid=True,
            kid=True,
            batch_size=args.eval_batch_size,
        )

        # log_dir = os.path.dirname(log)
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)
        with open(log, "a") as file:
            file.write(args.model_config_name_or_path + ", " + str(metrics) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
