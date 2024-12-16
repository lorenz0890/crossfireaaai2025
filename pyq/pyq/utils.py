from typing import Dict

import torch
from torch import Tensor
from torch.functional import F
from torch.nn import Linear, Module
from torch_geometric.nn import GCNConv


def equal_state_dictionary(state_dict_1: Dict[str, Tensor], state_dict_2: Dict[str, Tensor]):
    keys_1 = sorted(state_dict_1)
    keys_2 = sorted(state_dict_2)
    for key_1, key_2 in zip(*[keys_1, keys_2]):
        if torch.any(state_dict_1[key_1] != state_dict_2[key_2]):
            print([key_1])
            print([key_2])
            return False
    return True


class CustomLinear(Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_custom_layer = True

    def forward(self, x):
        return torch._C._nn.linear(x, self.weight, self.bias)


class GCN(Module):
    # source: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py#L29
    def __init__(self, in_channels, out_channels, intermediate_channels=16):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, intermediate_channels)
        self.conv2 = GCNConv(intermediate_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index = edge_index.long()
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ConvNet(Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNet, self).__init__()
        self.convs = torch.nn.Sequential(
            *[
                torch.nn.Conv2d(in_channels, 10, kernel_size=(5, 5)),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(10, 20, kernel_size=(5, 5)),
                torch.nn.MaxPool2d(2),
            ]
        )
        self.linars = torch.nn.Sequential(*[CustomLinear(320, 50), torch.nn.ReLU(), CustomLinear(50, out_channels)])

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, x.shape[1:].numel())
        x = self.linars(x)
        return F.softmax(x, dim=1)
