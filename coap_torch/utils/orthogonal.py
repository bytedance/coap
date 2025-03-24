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

@torch.no_grad()
def compute_orthogonal(in_matrix, rank, omega=None):
    """
    Args:
        in_matrix (torch.Tensor): the input matrix, [m, n]
        rank (int): the target rank of compression
        omage (torch.Tensor): if not provided, will generate a random matrix
    Returns:
        U, S, Vh (torch.Tensor)
    """
    device = in_matrix.device

    m, n = in_matrix.shape
    # Generate a random matrix
    if omega is None:
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            # TODO: use global random seed
            generator = torch.Generator(device=device).manual_seed(3407)
            omega = torch.randn([n, rank], device=device, generator=generator)
    if omega.data.dtype != torch.float:
        omega = omega.data.float().to(device)
    # Form sample matrix Y
    Y = in_matrix @ omega
    # Orthonormalize Y using QR decomposition
    Q, _ = torch.linalg.qr(Y, "reduced")  # [m, r]
    # Compute B = Q^T A
    B = Q.T @ in_matrix  # [r, n]

    # [r, r], [r], [r, n]
    U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)

    return Vh