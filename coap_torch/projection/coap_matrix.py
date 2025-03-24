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

from ..utils.orthogonal import compute_orthogonal


class Projector:
    """ A dummy projector class that does nothing """

    def __init__(self):
        pass

    def project(self, full_rank_grad, state):
        return full_rank_grad

    def project_back(self, low_rank_grad):
        return low_rank_grad


class MatrixCOAP:
    """
    A projector class for our algorithm.

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

    def __init__(
            self,
            rank,
            update_interval: int = 64,
            reproject_factor: int = 5,
            restore_state=False,
            scale=1.0,
    ):
        self.rank = rank
        self.update_interval = update_interval
        self.reproject_factor = reproject_factor
        self.restore_state = restore_state
        self.scale = scale
        self.ortho_matrix = None

    def project(self, full_rank_grad, state):
        return full_rank_grad

    @torch.no_grad()
    def update_proj(self, proj, proj_M, proj_type, grad, lr=0.1):
        """
        Update the projection matrix according to project-back error

        Args:
            proj (torch.Tensor): the projection matrix, [r, n]
            proj_M (torch.Tensor): the projected moment, [m, r]
            grad (torch.Tensor): the original gradient, [m, n]

        Returns:
            torch.Tensor: the updated projection matrix
        """

        exp_avg = None
        proj.requires_grad = True

        grad.requires_grad = False
        proj_M.requires_grad = False

        with torch.enable_grad():
            if proj_type == "right":
                exp_avg = proj_M @ proj
                loss = torch.nn.functional.mse_loss(grad @ proj.T @ proj, grad)
            else:
                exp_avg = proj @ proj_M
                loss = torch.nn.functional.mse_loss(proj @ proj.T @ grad, grad)
            cosine_sim = torch.nn.functional.cosine_similarity(exp_avg, grad, dim=1)
            loss *= 1 - cosine_sim.mean()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(proj, max_norm=1.0)
        proj -= proj.grad * lr
        proj.requires_grad = False
        if proj.grad is not None:
            proj.grad.zero_()
        return proj

    def restore(self, low_rank_grad):
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        return full_rank_grad

    def project_back(self, low_rank_grad):
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights
        if module_params.data.dtype != torch.float:
            float_data = False
            original_device = module_params.data.device
            original_type = module_params.data.dtype
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        omega = None
        if type == "right":
            if self.ortho_matrix is not None:
                omega = self.ortho_matrix.T
            Vh = compute_orthogonal(matrix, rank, omega)
            if not float_data:
                Vh = Vh.to(original_device).type(original_type)
            return Vh
        elif type == "left":
            if self.ortho_matrix is not None:
                omega = self.ortho_matrix
            Vh = compute_orthogonal(matrix.T, rank, omega)
            A = Vh.T  # [n, r]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        else:
            raise ValueError("type should be left, right or full")
