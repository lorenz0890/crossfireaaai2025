import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric

import torch_geometric.transforms as T
from torch.nn import BatchNorm1d, ModuleList
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.logging import log
from torch_geometric.nn import GCNConv, Aggregation, GINConv, MLP, JumpingKnowledge, BatchNorm
from torch_geometric.loader import DataLoader
from pyq.core.parser import TorchModelParser
from pyq.core.quantization.observer import UniformRangeObserver
from pyq.core.quantization.wrapper import ActivationQuantizerWrapper
from pyq.core.quantization.initializer import MinMaxInitializer, MinMaxOffsetInitializer
from pyq.core.quantization.functional import STEQuantizeFunction, STEOffsetQuantizeFunction
from pyq.core.quantization.wrapper import TransformationLayerQuantizerWrapper
from torch_geometric.nn import global_mean_pool, global_add_pool

dataset = "Cora"
hidden_channels = 32
lr = 0.001
epochs = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
#dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
#dataset = TUDataset(root='/tmp/COLLAB', name='COLLAB').shuffle() #transform=T.NormalizeFeatures()
#dataset = TUDataset(root='/tmp/DD', name='DD').shuffle()
dataset = TUDataset(root='/tmp/github_stargazers', name='github_stargazers').shuffle() #transform=T.NormalizeFeatures()
# Replace COLLAB with DD
# Colors-3
#exit()
#data = dataset[0]
#perm = torch.randperm(len(dataset))
#dataset = dataset[perm]
train_dataset = dataset[:int(len(dataset)*0.8)]
test_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
val_dataset = dataset[int(len(dataset)*0.9):]

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)
val_loader = DataLoader(val_dataset, batch_size=128)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GINConv(MLP([in_channels, hidden_channels, hidden_channels], dropout=0.5), train_eps=False)#GCNConv(in_channels, hidden_channels, cached=False, aggr='mean')
        self.conv2 = GINConv(MLP([hidden_channels, hidden_channels, hidden_channels], dropout=0.5),train_eps=False)  # GCNConv(in_channels, hidden_channels, cached=False, aggr='mean')
        self.conv3 = GINConv(MLP([hidden_channels, hidden_channels, hidden_channels], dropout=0.5),train_eps=False)  # GCNConv(in_channels, hidden_channels, cached=False, aggr='mean')
        self.conv4 = GINConv(MLP([hidden_channels, hidden_channels, hidden_channels], dropout=0.5),train_eps=False)  # GCNConv(in_channels, hidden_channels, cached=False, aggr='mean')
        self.conv5 = GINConv(MLP([hidden_channels, hidden_channels, out_channels], dropout=0.5, plain_last=True), train_eps=False)#GCNConv(hidden_channels, hidden_channels, cached=False, aggr='mean')
        #self.mlp = MLP([hidden_channels, hidden_channels, out_channels], norm=None, plain_last=True)
    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = global_add_pool(x, batch)
        #x = self.mlp(x)
        x = F.softmax(x, dim=1)
        return x


model = GCN(dataset.num_features, hidden_channels, dataset.num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#optimizer = torch.optim.Adam([
#    dict(params=model.conv1.parameters(), weight_decay=5e-4),
#    dict(params=model.conv2.parameters(), weight_decay=0)
#], lr=lr)  # Only perform weight-decay on first convolution.

#for data in train_loader:
#    print(data.edge_index.flatten().max())
#exit()
def train():
    model.train()

    total_loss = 0
    for data in train_loader:

        optimizer.zero_grad()
        data.to(device)
        x = torch.rand(data.num_nodes, 3).to(device)
        #x = torch_geometric.utils.degree(data.edge_index[0]).view(-1, 1)
        #print(data.num_nodes)
        #exit()
        out = model(x , data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss


@torch.no_grad()
def validate():
    model.eval()

    loaders = [train_loader, val_loader, test_loader]
    accs = []
    for i, loader in enumerate(loaders):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x = torch.rand(data.num_nodes, 3).to(device)
                #print(torch_geometric.utils.degree(data.edge_index.flatten()).shape)
                #x = torch_geometric.utils.degree(data.edge_index[0]).view(-1, 1)
                outputs = model(x , data.edge_index, data.batch)
                predicted = outputs.argmax(dim=-1)#torch.max(outputs.data, 1)
                total += data.y.size(0)
                correct += (predicted == data.y).sum().item()
                #if i == 2: print(predicted, data.y)

        accs.append(100 * correct / total)
    print(accs)
    return accs

if __name__ == '__main__':
    filename = "pyq_model_gin_gitstar.pth"  # TODO try again with new file name

    best_val_acc = final_test_acc = 0
    for epoch in range(1, epochs + 1):
        loss = train()
        train_acc, val_acc, tmp_test_acc = validate()
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)

    #exit()
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

    skip_layer_by_type = [GCNConv, Aggregation, BatchNorm, BatchNorm1d, GINConv, ModuleList, MLP]
    parser = TorchModelParser(
        callable_object=layer_quantizer,
        callable_object_kwargs=layer_quantizer_arguments,
        callable_object_for_nonparametric=activation_quantizer,
        callable_object_for_nonparametric_kwargs=activation_quantizer_arguments,
        skip_layer_by_type=skip_layer_by_type,
    )

    model = parser.apply(model)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, epochs+1):
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

