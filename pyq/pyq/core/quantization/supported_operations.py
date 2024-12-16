import torch
import torch_geometric

DEFAULT_OPERATION_NAME = "operation"
DEFAULT_ATTRIBUTES_NAME = "attribute"


layer_to_operation = {
    torch.nn.Conv2d: {
        DEFAULT_OPERATION_NAME: torch.nn.functional.conv2d,
        DEFAULT_ATTRIBUTES_NAME: ["bias", "stride", "padding", "dilation", "groups"],
    },
    torch.nn.modules.sparse.Embedding: {
        DEFAULT_OPERATION_NAME: torch.nn.functional.embedding,
        DEFAULT_ATTRIBUTES_NAME: ["padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse"],
    },
    torch.nn.Linear: {
        DEFAULT_OPERATION_NAME: torch.nn.functional.linear,
        DEFAULT_ATTRIBUTES_NAME: [
            "bias",
        ],
    },
    torch_geometric.nn.Linear: {
        DEFAULT_OPERATION_NAME: torch.nn.functional.linear,
        DEFAULT_ATTRIBUTES_NAME: [
            "bias",
        ],
    },
}
