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

from torch import nn
from typing import Iterable

MIN_CNN_PARAMS = 32*32*3*3  # A threshold controls the minimum tensor to project


def setup_params(self, params: Iterable[nn.parameter.Parameter]):
    """
    Setup the model's parameters for training with gradient projection

    Args:
        self (torch.optim.Optimizer):
            The Optimizer object.
        params (Iterable[nn.parameter.Parameter]):
            The model parameters to be optimzed.

    Returns:
        The parameter groups.
    """

    if self.proj_method is None:
        return params

    param_groups = list(params)
    if len(param_groups) == 0:
        raise ValueError("optimizer got an empty parameter list")
    if not isinstance(param_groups[0], dict):
        param_groups = [{"params": param_groups}]

    params_to_optimize = []

    for param_group in param_groups:
        params = param_group.pop("params")

        # Group params into different sets given the dim
        dim_1, dim_2, dim_4, dim_other = [], [], [], []

        for p in params:
            if p.requires_grad:
                if len(p.shape) == 2:
                    if self.rank is None:
                        dim_2.append(p)
                    elif self.rank >= min(p.shape):
                        dim_other.append(p)
                    else:
                        dim_2.append(p)
                elif len(p.shape) == 4:
                    if p.numel() <= MIN_CNN_PARAMS:
                        dim_other.append(p)
                    else:
                        dim_4.append(p)
                elif len(p.shape) == 1:
                    dim_1.append(p)
                else:
                    dim_other.append(p)

        params_to_optimize_in_group = [
            {"params": dim_1},
            {"params": dim_other},
            {
                "params": dim_2,
                "proj_method": self.proj_method,
                "restore_state": self.restore_state,
                "reproject_factor": self.reproject_factor,
                "rank_ratio_matrix": self.rank_ratio_matrix,
                "update_interval": self.update_interval,
                "rank": self.rank,
                "scale": self.scale
            },
            {
                "params": dim_4,
                "proj_method": self.proj_method,
                "rank_ratio_cnn": self.rank_ratio_cnn,
                "update_interval": self.update_interval,
                "reproject_factor": self.reproject_factor,
                "restore_state": self.restore_state,
                "scale": self.scale,
                "dim": 4,
            }
        ]

        for g in params_to_optimize_in_group:
            g.update(param_group)

        params_to_optimize.extend(params_to_optimize_in_group)

    return params_to_optimize
