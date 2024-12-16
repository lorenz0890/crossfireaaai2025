from typing import Any, Dict, List, Optional, Union

from torch import Tensor
from torch_geometric.nn import MLP


class MultiLayerPerceptron(MLP):
    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.0,
        act: str = "relu",
        batch_norm: bool = True,
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        batch_norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        relu_first: bool = False,
    ):
        super().__init__(
            channel_list=channel_list,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            batch_norm=batch_norm,
            act_first=act_first,
            act_kwargs=act_kwargs,
            batch_norm_kwargs=batch_norm_kwargs,
            bias=bias,
            relu_first=relu_first,
        )

    def forward(self, x: Tensor, edge_index: Tensor, batch) -> Tensor:
        return super(MultiLayerPerceptron, self).forward(x)
