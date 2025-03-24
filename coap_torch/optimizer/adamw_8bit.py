# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

""" Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py """

from typing import Iterable, Tuple, Callable

import math
import torch
import warnings
import bitsandbytes.functional as bnbf

from torch import nn
from torch.optim import Optimizer

from ..utils.params import setup_params
from ..projection.coap_matrix import MatrixCOAP, Projector
from ..projection.coap_high_order_tensor import TuckerCOAP

quant_fn = bnbf.quantize_blockwise
dequant_fn = bnbf.dequantize_blockwise


class COAPAdam8bitProjector(MatrixCOAP):
    """
    A 8-bit projector class for our projection algorithm.

    Args:
        rank (int):
            The rank of the projector.
        update_interval (int, optional, default to 200):
            Update interval to update projection matrix.
        reproject_factor (int, optional, default to 10):
            Frequency of recomputing proj matrix.
        is_adafactor (bool, optional, default to False):
            If used with adafactor.
        use_8bit (bool, optional, default to False):
            If used with 8bit.
        restore_state (bool, optional, default to True):
            If restore the state
        scale (float, optional, default to 1.0):
            The scaling factor for the projected gradients.
    """

    def project(self, full_rank_grad, state):
        iter = state["step"]

        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            _type = "right"
        else:
            _type = "left"

        if _type == "right":
            if self.ortho_matrix is None:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type=_type
                )

            elif iter % self.update_interval == 0:
                exp_avg = dequant_fn(*state["exp_avg"])
                if self.restore_state:
                    proj_exp_avg = self.restore(exp_avg)
                    exp_avg_sq = dequant_fn(*state["exp_avg_sq"])
                    proj_exp_avg_sq = self.restore(exp_avg_sq ** 0.5)
                # Perform projection matrix update
                if iter % (self.update_interval * self.reproject_factor) == 0:
                    # Re-compute the new projection
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type=_type
                    )
                else:
                    self.ortho_matrix = self.update_proj(
                        self.ortho_matrix,
                        exp_avg,
                        _type,
                        full_rank_grad,
                    )

                if self.restore_state:
                    state["exp_avg"] = quant_fn(torch.matmul(proj_exp_avg, self.ortho_matrix.T))
                    state["exp_avg_sq"] = quant_fn(torch.matmul(proj_exp_avg_sq, self.ortho_matrix.T) ** 2)
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.T)
        elif _type == "left":
            if self.ortho_matrix is None:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type=_type
                )
            elif iter % self.update_interval == 0:
                exp_avg = dequant_fn(*state["exp_avg"])
                if self.restore_state:
                    proj_exp_avg = self.restore(exp_avg)
                    exp_avg_sq = dequant_fn(*state["exp_avg_sq"])
                    proj_exp_avg_sq = self.restore(exp_avg_sq ** 0.5)
                if iter % (self.update_interval * self.reproject_factor) == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type=_type
                    )
                else:
                    self.ortho_matrix = self.update_proj(
                        self.ortho_matrix,
                        exp_avg,
                        _type,
                        full_rank_grad,
                    )
                if self.restore_state:
                    state["exp_avg"] = quant_fn(torch.matmul(self.ortho_matrix.T, proj_exp_avg))
                    state["exp_avg_sq"] = quant_fn(torch.matmul(self.ortho_matrix.T, proj_exp_avg_sq) ** 2)
            low_rank_grad = torch.matmul(self.ortho_matrix.T, full_rank_grad)
        return low_rank_grad


class AdamW8bit(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Args:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to `0.0`):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).

    Projection parameters:
        proj_method (`str`, *optional*, defaults to `None``):
            Specify the projection method, we support `adaptive` now, do nothing if is None.
        rank (`int`, *optional*, defaults to `None`):
            The rank of the projection space. Note that this can be overrided by `compression_ratio`.
            For a matrix with a shape of `(m, n)` (m >= n), the final projected would be `(m, r)`.
        rank_ratio (`float`, *optional*, defaults to `None`):
            Specify the compression ratio for projection, it will override rank if specified.
            The final rank would be calculated by `min(mat.shape) // compression_ratio`.

        project_cnn (`bool`, *optional*, defaults to `False`):
            If project the CNN layers with tucker.
        rank_ratio_cnn (`float`, *optional*, defaults to `None`):
            For CNN layers, we don't specify the rank for simplicity, due to the various CNN shapes.
            A compression ratio is required here to specify how much you want to compress.

        restore_state (`bool`, *optional*, defaults to `True`):
            If restore the states (moments) during training, emperically we set False for LLAMA.
        update_interval (`int`, *optional*, defaults to `30`):
            The interval of update the projection matrix with eq.5 from the paper, $T_{u}$.
        reproject_factor (`int`, *optional*, defaults to `10`):
            A factor of recompute the projection matrix, refer to $T$ and eq.6.
        scale (`float`, *optional*, defaults to `1.0`):
            The scaling factor for the projected gradients.
    """

    def __init__(
            self,
            params=None,
            grouped_params_with_proj=None,
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            no_deprecation_warning: bool = True,

            # Projection Args
            proj_method: str = 'coap',
            rank: int = None,
            rank_ratio_matrix: float = None,

            # Project CNN
            rank_ratio_cnn: float = None,

            restore_state: bool = False,
            update_interval: int = 32,
            reproject_factor: int = 5,
            scale: float = 1.0
    ):
        self.proj_method = proj_method
        self.rank = rank
        self.rank_ratio_matrix = rank_ratio_matrix

        self.rank_ratio_cnn = rank_ratio_cnn

        self.restore_state = restore_state
        self.update_interval = update_interval
        self.reproject_factor = reproject_factor
        self.scale = scale

        if grouped_params_with_proj is None:
            assert params is not None
            grouped_params_with_proj = setup_params(self, params)

        # Sanity checks
        if self.rank is not None and self.rank_ratio_matrix is not None:
            warnings.warn(
                "You have specified a compression ratio for projection, which will override `rank`"
            )

        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )

        # require_version("torch>=1.5.0")  # add_ with alpha

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }

        super().__init__(grouped_params_with_proj, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # Projection
                if "proj_method" in group:
                    # Initialize projector
                    if "projector" not in state:
                        if len(p.shape) == 2:
                            # NOTE: compression_ratio has the highest priority
                            rank_ratio = group.get("rank_ratio_matrix", self.rank_ratio_matrix)
                            if rank_ratio is not None:
                                rank = max(int(min(grad.shape) // rank_ratio), 1)
                            else:
                                rank = min(group["rank"], min(grad.shape))
                            if group["proj_method"] == "coap":
                                state["projector"] = COAPAdam8bitProjector(
                                    rank,
                                    restore_state=group.get("restore_state", self.restore_state),
                                    update_interval=group.get("update_interval", self.update_interval),
                                    reproject_factor=group.get("reproject_factor", self.reproject_factor),
                                    scale=group.get("scale", self.scale),
                                )
                            else:
                                raise NotImplementedError(f"Unsupported projection type: {group['proj_method']}")
                        elif len(p.shape) == 4:
                            state["projector"] = TuckerCOAP(
                                rank_ratio=group.get("rank_ratio_cnn", self.rank_ratio_cnn),
                                restore_state=group.get("restore_state", self.restore_state),
                                update_interval=group.get("update_interval", self.update_interval),
                                reproject_factor=group.get("reproject_factor", self.reproject_factor),
                                scale=group.get("scale", self.scale),
                                use_8bit=True,
                            )
                        elif len(p.shape) < 2:
                            state["projector"] = Projector()
                        else:
                            raise NotImplementedError(f"Unsupported input shape: {grad.shape}")
                    grad = state["projector"].project(grad, state)

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = quant_fn(torch.zeros_like(grad))
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = quant_fn(torch.zeros_like(grad))

                exp_avg = dequant_fn(*state["exp_avg"])
                exp_avg_sq = dequant_fn(*state["exp_avg_sq"])
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                            step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                # Norm gradient
                norm_grad = exp_avg / denom

                # Project Back
                if "proj_method" in group:
                    norm_grad = state["projector"].project_back(norm_grad)
                state["exp_avg"] = quant_fn(exp_avg)
                state["exp_avg_sq"] = quant_fn(exp_avg_sq)
                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss