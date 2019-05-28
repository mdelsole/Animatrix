from torch import nn
import torch


class v4Cell(nn.Module):
    def forward(self, s2_outputs):
        # Take max value of the inflowing v2 cells
        maxs = [s2.max(dim=3)[0] for s2 in s2_outputs]
        maxs = [m.max(dim=2)[0] for m in maxs]
        maxs = torch.cat([m[:, None, :] for m in maxs], 1)

        return maxs.max(dim=1)[0]