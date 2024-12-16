from typing import Any, Optional

from torch_geometric.nn import GAT

from pyq.models.graph.base_mpnn import BaseMessagePassing


class GraphAttentionNetwork(BaseMessagePassing):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 1,
        module_use_atom_encoder: bool = False,
        module_use_mlp_output: bool = False,
        module_global_pool: Optional[str] = None,
        module_dropout: float = 0.5,
        **kwargs: Any,
    ):
        super(GraphAttentionNetwork, self).__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            module_use_atom_encoder=module_use_atom_encoder,
            module_use_mlp_output=module_use_mlp_output,
            module_global_pool=module_global_pool,
            module_dropout=module_dropout,
        )
        assert num_layers >= 1, "number of layers suppose to be greater than `1`"

        if module_use_mlp_output:
            out_channels = hidden_channels

        self.convs = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            **kwargs,
        )
