from typing import Optional

from ogb.graphproppred.mol_encoder import AtomEncoder
from torch.nn import Dropout, Linear, Module, ModuleDict, ReLU
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


class BaseMessagePassing(Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        module_use_atom_encoder: bool = False,
        module_use_mlp_output: bool = False,
        module_global_pool: Optional[str] = None,
        module_dropout: float = 0.5,
    ):
        super(BaseMessagePassing, self).__init__()
        assert module_global_pool in [
            None,
            "add",
            "max",
            "mean",
        ], "{} is not implemented, available global_pool 'add', 'max', 'mean''".format(module_global_pool)

        self.use_atom_encoder = module_use_atom_encoder
        self.use_mlp = module_use_mlp_output
        self.global_pool_function = module_global_pool
        if module_global_pool == "add":
            self.global_pool_function = global_add_pool
        if module_global_pool == "max":
            self.global_pool_function = global_max_pool
        if module_global_pool == "mean":
            self.global_pool_function = global_mean_pool

        if module_use_atom_encoder:
            self.atom_encoder = AtomEncoder(in_channels)

        self.convs = None
        self.mlp = ModuleDict()

        if module_use_mlp_output:
            self.mlp.update(
                {
                    # first linear layer
                    "linear_0": Linear(hidden_channels, hidden_channels, bias=False),
                    "relu_0": ReLU(),
                    "dropout_0": Dropout(module_dropout),
                    # second linear layer
                    "linear_1": Linear(hidden_channels, out_channels, bias=False),
                }
            )

    def reset_parameters(self):
        self.convs.reset_parameters()
        for name, layer in self.mlp.items():
            layer.reset_parameters()

    # TODO (Samir): update the arguments in forward signature to accept `edge_weight`, and `edge_attr`
    def forward(self, x, edge_index, batch):
        if self.convs is None:
            raise ValueError("initialize the `convs` for the message passing module.")

        if self.use_atom_encoder:
            x = self.atom_encoder(x)

        x = self.convs(x, edge_index)

        if self.global_pool_function:
            x = self.global_pool_function(x, batch)
        for name, layer in self.mlp.items():
            x = layer(x)
        return x
