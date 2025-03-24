# Copyright (C) 2025 ByteDance
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

import torch


def get_size(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size(), value.numel()
    elif isinstance(value, tuple):
        return sum(get_size(v)[0] for v in value), sum(get_size(v)[1] for v in value)
    elif isinstance(value, dict):
        return sum(get_size(v)[0] for v in value.values()), sum(
            get_size(v)[1] for v in value.values()
        )
    elif isinstance(value, list):
        return sum(get_size(v)[0] for v in value), sum(get_size(v)[1] for v in value)
    elif hasattr(value, "ortho_matrix"):
        proj = value.ortho_matrix
        return sum(get_size(v)[0] for v in proj), sum(get_size(v)[1] for v in proj)
    else:
        return 0, 0


def show_memory_usage(model, optimizer, optimizer_ckpt=None):
    if getattr(show_memory_usage, 'executed', False):
        return
    show_memory_usage.executed = True

    total_params = 0
    total_size = 0
    state_dict = None

    if optimizer is not None:
        state_dict = optimizer.state_dict()
    elif optimizer_ckpt is not None:
        state_dict = optimizer_ckpt

    if "state" in state_dict:
        for state in state_dict["state"].values():
            size, params = get_size(state)
            total_size += size
            total_params += params
        for state in state_dict["param_groups"]:
            size, params = get_size(state)
            total_size += size
            total_params += params
        print(f"[COAP] Total parameters in optimizer: {total_params}")
        print(f"[COAP] Total size in optimizer: {total_size / (1024 * 1024):.2f} MB")

    if model is not None:
        total_params = 0
        total_size = 0
        for state in model.parameters():
            size, params = get_size(state)
            total_size += size
            total_params += params
        print(f"[COAP] Total parameters in model: {total_params}")
        print(f"[COAP] Total size in model: {total_size / (1024 * 1024):.2f} MB")
        print(torch.cuda.memory_summary())