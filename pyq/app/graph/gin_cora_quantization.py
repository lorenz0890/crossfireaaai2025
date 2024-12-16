import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import log
from torch_geometric.nn import GCNConv, Aggregation
from torch_geometric.nn import GINConv
from torch.nn import Sequential

from pyq.core.parser import TorchModelParser
from pyq.core.quantization.observer import UniformRangeObserver
from pyq.core.quantization.wrapper import ActivationQuantizerWrapper
from pyq.core.quantization.initializer import MinMaxInitializer, MinMaxOffsetInitializer
from pyq.core.quantization.functional import STEQuantizeFunction, STEOffsetQuantizeFunction
from pyq.core.quantization.wrapper import TransformationLayerQuantizerWrapper

dataset = "PubMed"
hidden_channels = 30
lr = 0.01
epochs = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

'''
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.conv4 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.conv5 = GCNConv(hidden_channels, out_channels, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv5(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)#
'''

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # Define an MLP for use in GINConv
        # Adjust the MLP architecture as needed
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )

        # Define GINConv layers using the specified MLP
        self.conv1 = GINConv(mlp)

        # For subsequent GINConv layers, you might need different MLPs if you want different architectures
        # Here, reusing the `mlp` architecture for simplicity; consider defining new MLPs for each GINConv layer
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        ))
        self.conv3 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        ))
       # self.conv4 = GINConv(torch.nn.Sequential(
       #     torch.nn.Linear(hidden_channels, hidden_channels),
       #     torch.nn.ReLU(),
       #     torch.nn.Linear(hidden_channels, hidden_channels),
       # ))
       # self.conv5 = GINConv(torch.nn.Sequential(
       #     torch.nn.Linear(hidden_channels, out_channels),
       #     torch.nn.ReLU(),  # Optional: consider whether you need an activation before the final layer
       # ))

    def forward(self, x, edge_index, edge_weight=None):
        # Note: GINConv does not use edge_weight, so it's omitted in convolutions
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index).relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv4(x, edge_index).relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv5(x, edge_index)

        return F.log_softmax(x, dim=1)
#
model = GIN(dataset.num_features, hidden_channels, dataset.num_classes)
model, data = model.to(device), data.to(device)
#optimizer = torch.optim.Adam([
#    dict(params=model.conv1.parameters(), weight_decay=5e-4),
#    dict(params=model.conv2.parameters(), weight_decay=5e-4),
#    dict(params=model.conv3.parameters(), weight_decay=5e-4),
#    dict(params=model.conv4.parameters(), weight_decay=5e-4),
#    dict(params=model.conv5.parameters(), weight_decay=5e-4)
#], lr=lr)  # Only perform weight-decay on first convolution.
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def validate():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


if __name__ == '__main__':
    filename = "pyq_model_gin_pubmed.pth"  # TODO try again with new file name

    best_val_acc = final_test_acc = 0
    for epoch in range(1, epochs + 1):
        loss = train()
        train_acc, val_acc, tmp_test_acc = validate()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)


    bits = 8
    # !important: configure the quantization function and initialization
    layer_quantizer = TransformationLayerQuantizerWrapper
    layer_quantizer_arguments = {
        "quantizer_function": STEQuantizeFunction(),
        "initializer": MinMaxInitializer().to(device),
        "range_observer": UniformRangeObserver(bits=bits).to(device),
    }

    activation_quantizer = ActivationQuantizerWrapper
    activation_quantizer_arguments = {
        "quantizer_function": STEOffsetQuantizeFunction(),
        "initializer": MinMaxOffsetInitializer().to(device),
        "range_observer": UniformRangeObserver(bits=bits, is_positive=True).to(device),
    }

    skip_layer_by_type = [GINConv, GCNConv, Aggregation, Sequential]
    parser = TorchModelParser(
        callable_object=layer_quantizer,
        callable_object_kwargs=layer_quantizer_arguments,
        callable_object_for_nonparametric=activation_quantizer,
        callable_object_for_nonparametric_kwargs=activation_quantizer_arguments,
        skip_layer_by_type=skip_layer_by_type,
    )

    model = parser.apply(model)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, epochs + 1):
        loss = train()
        train_acc, val_acc, tmp_test_acc = validate()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)

    unparser = TorchModelParser(
        callable_object=layer_quantizer.unwrap,
        callable_object_for_nonparametric=activation_quantizer.unwrap,
        skip_layer_by_type=skip_layer_by_type,
    )

    final_model = model  # unparser.apply(model)

    if not filename == "":
        torch.save(
            {
                "Val": best_val_acc,
                'Model': final_model
            },
            filename,
        )

