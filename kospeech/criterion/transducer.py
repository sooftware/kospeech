# Copyright (c) 2021, Soohwan Kim. All rights reserved.
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
import torch.nn as nn


class TransducerLoss(nn.Module):
    """
    Transducer loss module.

    Args:
        blank_id (int): blank symbol id
    """

    def __init__(self, blank_id: int) -> None:
        """Construct an TransLoss object."""
        super().__init__()
        try:
            from warp_rnnt import rnnt_loss
        except ImportError:
            raise ImportError("warp-rnnt is not installed. Please re-setup")
        self.rnnt_loss = rnnt_loss
        self.blank_id = blank_id

    def forward(
            self,
            log_probs: torch.FloatTensor,
            targets: torch.IntTensor,
            input_lengths: torch.IntTensor,
            target_lengths: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Compute path-aware regularization transducer loss.

        Args:
            log_probs (torch.FloatTensor): Batch of predicted sequences (batch, maxlen_in, maxlen_out+1, odim)
            targets (torch.IntTensor): Batch of target sequences (batch, maxlen_out)
            input_lengths (torch.IntTensor): batch of lengths of predicted sequences (batch)
            target_lengths (torch.IntTensor): batch of lengths of target sequences (batch)

        Returns:
            loss (torch.FloatTensor): transducer loss
        """

        return self.rnnt_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction="mean",
            blank=self.blank_id,
            gather=True,
        )
