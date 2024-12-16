import copy
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import ModuleList, BatchNorm1d, Sequential, Embedding

from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GlobalAttention, MessagePassing, Set2Set, global_add_pool, global_max_pool,
                                global_mean_pool, Aggregation, BatchNorm)
from torch_geometric.utils import degree
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


from pyq.core.parser import TorchModelParser
from pyq.core.quantization.observer import UniformRangeObserver
from pyq.core.quantization.initializer import MinMaxInitializer, MinMaxOffsetInitializer
from pyq.core.quantization.functional import STEQuantizeFunction, STEOffsetQuantizeFunction
from pyq.core.quantization.wrapper import ActivationQuantizerWrapper, TransformationLayerQuantizerWrapper

import torch
import torch_geometric
from torch_geometric.nn import GCNConv, GINConv, MLP
from torch_geometric.data import Data, Batch, DataLoader, Dataset
import random
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool, global_add_pool
import os.path as osp

# skip warnings
warnings.filterwarnings("ignore")

class GeneratedNodeClassificationDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return ["data_{}.pt".format(i) for i in range(0,1300)]


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def get_idx_split(self):
        return {'train' : [i for i in range(0, 975)],
                'test' : [i for i in range(975, 1300)]
                }

class GNN(torch.nn.Module):
    def __init__(self, d, M, C):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.d = d
        mlp = MLP(in_channels=M, hidden_channels=M * 10, out_channels=M * 10, num_layers=3)
        self.convs.append(GINConv(mlp, eps=-1.0, train_eps=False))  # GCNConv(3, 3)
        for i in range(self.d - 1):
            mlp = MLP(in_channels=M*10, hidden_channels=M*10, out_channels=M*10, num_layers=3)
            self.convs.append(GINConv(mlp, eps=-1.0, train_eps=False))  # GCNConv(3, 3)

        mlp = MLP(in_channels=M*10, hidden_channels=M*10, out_channels=C, num_layers=3)
        self.convs.append(GINConv(mlp, eps=-1.0, train_eps=False))  # GCNConv(3, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.d - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, training=self.training)

        x = self.convs[self.d - 1](x, edge_index)
        # x = F.relu(x)

        return  F.log_softmax(x, dim=1)

cls_criterion = F.cross_entropy#torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            #print(batch, flush=True)
            # ignore nan targets (unlabeled) when computing training loss.
            if "classification" in task_type:
                loss = cls_criterion(pred, batch.y)
            else:
                loss = reg_criterion(pred, batch.y)
            loss.backward()
            optimizer.step()

def eval(model, device, loader):
    model.eval()
    acc, ctr = 0, 0
    for batch in loader:
        # print(batch, batch.num_graphs)
        ctr += 1
        pred = model(batch.to(device)).argmax(dim=1)
        # print(pred, '\n', batch.y)
        # break
        correct = (pred == batch.y).sum()
        # print(int(correct) / int(batch.y.shape[0]))
        acc += int(correct) / int(batch.y.shape[0])
    model.train()
    return acc/ctr


def main():
    # Training settings
    device = 0
    dataset_name = "node_classification"
    batch_size = 32
    num_layer = 5
    emb_dim = 300
    drop_ratio = 0.5
    epochs = 20
    num_workers = 12
    feature = "full"
    filename = "pyq_model_gin_node_int2.pth" #TODO try again with new file name

    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")

    # automatic dataloading and splitting
    dataset = GeneratedNodeClassificationDataset('/media/lorenz/Volume/code/pyq_main/pyq/app/graph/dataset/node_classification_small/')


    split_idx = dataset.get_idx_split()

    # automatic evaluator. takes dataset name as input
    #evaluator = Evaluator(dataset_name)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    #valid_loader = DataLoader(
    #    dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False, num_workers=num_workers
    #)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = GNN(d=4,
                M=1,
                C=8
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    valid_curve = []
    test_curve = []
    train_curve = []

    torch.save({'Model': model}, './temp_model.pth')
    best_test_acc = 0.0
    train_float32 = True
    if train_float32:
        for epoch in range(1, epochs + 1):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            train(model, device, train_loader, optimizer, 'classification')


            if epoch % 1 == 0 or epoch == epochs - 1:
                print("Evaluating...")
                test_acc = eval(model, device, test_loader)
                train_acc = eval(model, device, train_loader)
                print(epoch, f'Test: {test_acc:.4f}', f'Train: {train_acc:.4f}', )

                if test_acc > best_test_acc:
                    print('update')
                    best_test_acc = test_acc
                    torch.save({'Model': model}, './temp_model.pth')

                test_curve.append(test_acc)
                train_curve.append(train_acc)


    bits = 2
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

    skip_layer_by_type = [Sequential, ModuleList, BondEncoder, BatchNorm1d, BatchNorm, AtomEncoder, Aggregation, GINConv, Embedding, MLP]
    parser = TorchModelParser(
        callable_object=layer_quantizer,
        callable_object_kwargs=layer_quantizer_arguments,
        callable_object_for_nonparametric=activation_quantizer,
        callable_object_for_nonparametric_kwargs=activation_quantizer_arguments,
        skip_layer_by_type=skip_layer_by_type,
    )

    unparser = TorchModelParser(
        callable_object=layer_quantizer.unwrap,
        callable_object_for_nonparametric=activation_quantizer.unwrap,
        skip_layer_by_type=skip_layer_by_type,
    )

    model = parser.apply(torch.load("./temp_model.pth")['Model'])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    valid_curve = []
    test_curve = []
    train_curve = []

    best_model = copy.deepcopy(model)
    best_test_acc = 0.0
    for epoch in range(1, epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train(model, device, train_loader, optimizer, 'classification')


        if epoch % 1 == 0 or epoch == epochs - 1:
            print("Evaluating...")
            test_acc = eval(model, device, test_loader)
            train_acc = eval(model, device, train_loader)
            print(epoch, f'Test: {test_acc:.4f}', f'Train: {train_acc:.4f}', )
            # print(out.shape, batch.y.shape)
            # print(epoch, loss.item())
            if test_acc > best_test_acc:
                print('update')
                best_test_acc = test_acc
                torch.save({'Model' : model}, './temp_model.pth')

            test_curve.append(test_acc)
            train_curve.append(train_acc)

    if "classification" in 'classification':#dataset.task_type:
        best_test_epoch = np.argmax(np.array(test_curve))
        best_train = max(train_curve)
    #else:
    #    best_val_epoch = np.argmin(np.array(valid_curve))
    #    best_train = min(train_curve)

    final_model = torch.load("./temp_model.pth")['Model']  # best_model#model  # unparser.apply(model)
    print(eval(final_model.to(device), device, test_loader), flush=True)
    print("Finished training!")
    #print("Best validation score: {}".format(valid_curve[best_val_epoch]))
    #print("Test score: {}".format(test_curve[best_val_epoch]))

    #print(final_model._wrapped_object._wrapped_object.gnn_node.convs[0].mlp[3]._wrapped_object.weight, flush=True)
    if not filename == "":
        torch.save(
            {
    #            "Val": valid_curve[best_val_epoch],
                "Test": test_curve[best_test_epoch],
                "Train": train_curve[best_test_epoch],
                "BestTrain": best_train,
                'Model':final_model
            },
            filename,
        )
if __name__ == "__main__":
    main()
