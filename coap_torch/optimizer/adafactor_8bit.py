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

from typing import Iterable, Tuple

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


class COAPAdafactor8bitProjector(MatrixCOAP):
    """
    A projector class for our adaptive projection algorithm.

    Args:
        rank (int):
            The rank of the projector.
        update_interval (int, optional, default to 200):
            Update interval to update projection matrix.
        reproject_factor (int, optional, default to 10):
            Frequency of recomputing proj matrix.
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
                exp_avg = state["exp_avg"]
                if self.restore_state:
                    exp_avg = dequant_fn(*state["exp_avg"])
                    proj_exp_avg = self.restore(exp_avg)
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
                    state["exp_avg"] = torch.matmul(proj_exp_avg, self.ortho_matrix.T)
                    state["exp_avg"] = quant_fn(state["exp_avg"])
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.T)
        elif _type == "left":
            if self.ortho_matrix is None:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type=_type
                )
            elif iter % self.update_interval == 0:
                exp_avg = state["exp_avg"]
                if self.restore_state:
                    exp_avg = dequant_fn(*state["exp_avg"])
                    proj_exp_avg = self.restore(exp_avg)
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
                    state["exp_avg"] = torch.matmul(self.ortho_matrix.T, proj_exp_avg)
                    state["exp_avg"] = quant_fn(state["exp_avg"])
            low_rank_grad = torch.matmul(self.ortho_matrix.T, full_rank_grad)
        return low_rank_grad


class Adafactor8bit(Optimizer):
    """
    AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code:
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

    Paper: *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* https://arxiv.org/abs/1804.04235 Note that
    this optimizer internally adjusts the learning rate depending on the `scale_parameter`, `relative_step` and
    `warmup_init` options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Args:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*):
            The external learning rate.
        eps (`Tuple[float, float]`, *optional*, defaults to `(1e-30, 0.001)`):
            Regularization constants for square gradient and parameter scale respectively
        clip_threshold (`float`, *optional*, defaults to 1.0):
            Threshold of root mean square of final gradient update
        decay_rate (`float`, *optional*, defaults to -0.8):
            Coefficient used to compute running averages of square
        beta1 (`float`, *optional*):
            Coefficient used for computing running averages of gradient
        weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay (L2 penalty)
        scale_parameter (`bool`, *optional*, defaults to `True`):
            If True, learning rate is scaled by root mean square
        relative_step (`bool`, *optional*, defaults to `True`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (`bool`, *optional*, defaults to `False`):
            Time-dependent learning rate computation depends on whether warm-up initialization is being used

    Projection parameters:
        proj_method (`str`, *optional*, defaults to `None``):
            Specify the projection method, we support `adaptive` now, do nothing if is None.
        rank (`int`, *optional*, defaults to `None`):
            The rank of the projection space. Note that this can be overrided by `compression_ratio`.
            For a matrix with a shape of `(m, n)` (m >= n), the final projected would be `(m, r)`.
        rank_ratio_matrix (`float`, *optional*, defaults to `None`):
            Specify the compression ratio for projection, it will override rank if specified.
            The final rank would be calculated by `min(mat.shape) // rank_ratio_matrix`.

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
            lr: float = None,
            eps: Tuple[float, float] = (1e-30, 1e-3),
            clip_threshold: float = 1.0,
            decay_rate: float = -0.8,
            beta1: float = None,
            weight_decay: float = 0.0,
            scale_parameter: bool = True,
            relative_step: bool = False,
            warmup_init: bool = False,

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

        # require_version("torch>=1.5.0")  # add_ with alpha

        if lr is not None and relative_step:
            raise ValueError(
                "Cannot combine manual `lr` and `relative_step=True` options"
            )
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        }

        super().__init__(grouped_params_with_proj, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): 
                A closure that reevaluates the model and returns the loss.
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

                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                if "dim" not in group:
                    group["dim"] = 2

                # COAP Projection
                if "proj_method" in group:
                    # Initialize projector
                    if "projector" not in state:
                        if len(p.shape) == 2:
                            rank_ratio = group.get("rank_ratio_matrix", self.rank_ratio_matrix)
                            if rank_ratio is not None:
                                rank = max(int(min(grad.shape) // rank_ratio), 1)
                            else:
                                rank = min(group["rank"], min(grad.shape))
                            if group["proj_method"] == "coap":
                                state["projector"] = COAPAdafactor8bitProjector(
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
                                is_adafactor=True,
                            )
                        elif len(p.shape) < 2:
                            state["projector"] = Projector()
                        else:
                            raise NotImplementedError(f"Unsupported input shape: {grad.shape}")

                    grad = state["projector"].project(grad, state)

                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)

                # State Initialization
                if "RMS" not in state:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = quant_fn(torch.zeros_like(grad))
                    if factored:
                        state["exp_avg_sq_row"] = quant_fn(torch.zeros(grad_shape[:-1]).to(grad))
                        state["exp_avg_sq_col"] = quant_fn(torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).to(grad))
                    else:
                        state["exp_avg_sq"] = quant_fn(torch.zeros_like(grad))

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"]
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"]
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"]
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"]

                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad ** 2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = dequant_fn(*state["exp_avg_sq_row"])
                    exp_avg_sq_col = dequant_fn(*state["exp_avg_sq_col"])

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=(1.0 - beta2t)
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=(1.0 - beta2t)
                    )

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = dequant_fn(*state["exp_avg_sq"])

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )
                update.mul_(lr)
                if use_first_moment:
                    exp_avg = dequant_fn(*state["exp_avg"])
                    exp_avg.mul_(group["beta1"]).add_(
                        update, alpha=(1 - group["beta1"])
                    )
                    update = exp_avg

                # Project Back
                if "proj_method" in group:
                    update = state["projector"].project_back(update)

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))
                p_data_fp32.add_(-update)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)
                if "rank" in group:
                    state["exp_avg"] = quant_fn(exp_avg)
                    if factored:
                        state["exp_avg_sq_row"] = quant_fn(exp_avg_sq_row)
                        state["exp_avg_sq_col"] = quant_fn(exp_avg_sq_col)
                    else:
                        state["exp_avg_sq"] = quant_fn(exp_avg)
        return loss

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = (
                1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)