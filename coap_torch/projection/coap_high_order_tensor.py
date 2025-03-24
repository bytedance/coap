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
import tensorly as tl
import bitsandbytes.functional as bnbf

from tensorly.decomposition import partial_tucker

tl.set_backend("pytorch")

quant_fn = bnbf.quantize_blockwise
dequant_fn = bnbf.dequantize_blockwise


class TuckerCOAP:
    """
    A projector class for our adaptive projection with turcker.

    Args:
        rank (int, optional, default to None): 
            The rank of the projector.
        update_interval (int, optional, default to 200): 
            Update interval to update projection matrix.
        scale (float, optional, default to 1.0): 
            The scaling factor for the projected gradients.
        rank_ratio (float, optional, default to 2.0): 
            Compression rate.
        is_adafactor (bool, optional): 
            If used with adafactor.
        use_8bit (bool, optional, default to False): 
            If used with 8bit.
        restore_state (bool, optional, default to True): 
            If restore the state
        reproject_factor (int, optional, default to 10): 
            Frequency of recomputing proj matrix.
    """

    def __init__(
            self,
            rank=None,
            update_interval=32,
            scale=1.0,
            rank_ratio=2.0,
            is_adafactor=False,
            use_8bit=False,
            restore_state=False,
            reproject_factor=5,
    ):
        self.rank = rank
        self.update_interval = update_interval
        self.scale = scale
        self.ortho_matrix = None
        self.rank_ratio = rank_ratio
        self.is_adafactor = is_adafactor
        self.use_8bit = use_8bit
        self.restore_state = restore_state
        self.reproject_factor = reproject_factor

    def project(self, full_rank_grad, state):
        """
        Projects the full-rank gradients onto the low-rank subspace.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            state: parameter state from optimizer.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        if self.rank is None:
            self.rank = []
            if full_rank_grad.shape[-1] == 1:
                self.rank.append(int(max(1, full_rank_grad.shape[0] // self.rank_ratio)))
            else:
                for s in full_rank_grad.shape[:2]:
                    self.rank.append(int(max(1, s // self.rank_ratio ** 0.5)))

        iter = state["step"]

        if self.ortho_matrix is None:
            self.ortho_matrix = [None] * len(self.rank)
            self.ortho_matrix = self.get_coap_orthogonal_matrix(full_rank_grad)

        elif iter % self.update_interval == 0:
            exp_avg = state["exp_avg"]
            if self.restore_state:
                if self.use_8bit:
                    exp_avg = dequant_fn(*state["exp_avg"])
                proj_exp_avg = self.restore(exp_avg)
                if not self.is_adafactor:
                    exp_avg_sq = state["exp_avg_sq"]
                    if self.use_8bit:
                        exp_avg_sq = dequant_fn(*state["exp_avg_sq"])
                    proj_exp_avg_sq = self.restore(exp_avg_sq ** 0.5)

            if iter % (self.update_interval * self.reproject_factor) == 0:
                self.ortho_matrix = self.get_coap_orthogonal_matrix(full_rank_grad)
            else:
                self.update_proj(exp_avg, full_rank_grad)

            if self.restore_state:
                state["exp_avg"] = self.transform(self.ortho_matrix, proj_exp_avg)
                if self.use_8bit:
                    state["exp_avg"] = quant_fn(state["exp_avg"])
                if not self.is_adafactor:
                    state["exp_avg_sq"] = (
                            self.transform(self.ortho_matrix, proj_exp_avg_sq) ** 2
                    )
                    if self.use_8bit:
                        state["exp_avg_sq"] = quant_fn(state["exp_avg_sq"])

        transformed_low_rank = self.transform(self.ortho_matrix, full_rank_grad)

        return transformed_low_rank

    @torch.no_grad()
    def update_proj(self, proj_exp_avg, grad, lr=0.1):
        exp_avg = None
        for i in range(len(self.rank)):
            self.ortho_matrix[i].requires_grad = True

        proj_exp_avg.requires_grad = False
        grad.requires_grad = False

        with torch.enable_grad():
            if len(self.rank) == 2:
                exp_avg = torch.einsum(
                    "mncd, am, bn->abcd",
                    proj_exp_avg,
                    self.ortho_matrix[0],
                    self.ortho_matrix[1],
                )
                proj_grad = torch.einsum(
                    "abcd, am, bn->mncd",
                    grad,
                    self.ortho_matrix[0],
                    self.ortho_matrix[1],
                )
                proj_grad = torch.einsum(
                    "mncd, am, bn->abcd",
                    proj_grad,
                    self.ortho_matrix[0],
                    self.ortho_matrix[1],
                )
            else:
                exp_avg = torch.einsum(
                    "mbcd, am->abcd", proj_exp_avg, self.ortho_matrix[0]
                )
                proj_grad = torch.einsum("abcd, am->mbcd", grad, self.ortho_matrix[0])
                proj_grad = torch.einsum(
                    "mbcd, am->abcd", proj_grad, self.ortho_matrix[0]
                )

            loss = torch.nn.functional.mse_loss(proj_grad, grad)
            cosine_sim = torch.nn.functional.cosine_similarity(exp_avg, grad, dim=1)
            loss *= 1 - cosine_sim.mean()

            loss.backward()

        for i in range(len(self.rank)):
            torch.nn.utils.clip_grad_norm_(self.ortho_matrix[i], max_norm=1.0)
            self.ortho_matrix[i] -= self.ortho_matrix[i].grad * lr
            self.ortho_matrix[i].requires_grad = False

    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradients back to the full-rank space.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The full-rank gradients.
        """
        full_rank_grad = self.inverse_transform(self.ortho_matrix, low_rank_grad)
        return full_rank_grad * self.scale

    def restore(self, low_rank_grad):
        """
        Projects the low-rank gradients back to the full-rank space.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The full-rank gradients.
        """
        full_rank_grad = self.inverse_transform(self.ortho_matrix, low_rank_grad)
        return full_rank_grad

    def get_coap_orthogonal_matrix(self, weights):
        module_params = weights
        if module_params.data.dtype != torch.float:
            matrix = module_params.data.float()
        else:
            matrix = module_params.data
        if len(self.rank) == 2:
            core, _ = partial_tucker(matrix, modes=[0, 1], rank=self.rank)
            return core[1]
        core, _ = partial_tucker(matrix, modes=[0], rank=self.rank)
        return core[1]

    def transform(self, tensor, x):
        """
        Transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        if len(self.rank) == 2:
            return torch.einsum("abcd, am, bn->mncd", x, tensor[0], tensor[1])
        return torch.einsum("abcd, am->mbcd", x, tensor[0])

    def inverse_transform(self, tensor, x):
        """
        Inverse transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The inverse transformed tensor.
        """
        if len(self.rank) == 2:
            return torch.einsum("mncd, am, bn->abcd", x, tensor[0], tensor[1])
        return torch.einsum("mbcd, am->abcd", x, tensor[0])
