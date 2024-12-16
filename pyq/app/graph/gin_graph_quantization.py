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
import copy

# skip warnings
warnings.filterwarnings("ignore")

class GeneratedGraphClassificationDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return ["data_{}.pt".format(i) for i in range(0,6000)]
        #return ["data_{}.pt".format(i) for i in range(0, 20000)]


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def get_idx_split(self):
        return {'train' : [i for i in range(0, 4020)],
                'test' : [i for i in range(4020, 6000)]
                }
        #return {'train': [i for i in range(0, 15000)],
        #        'test' : [i for i in range(15000, 20000)]
        #        }


class GNN(torch.nn.Module):
    def __init__(self, d, M, C):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.d = d
        # Interesting: if we keep M at 1 it is still able to learn the task but cant be quantized w/o loosing accuracy!
        mlp = MLP(in_channels=M, hidden_channels=M * 100, out_channels=M * 100, num_layers=3)
        # self.convs.append(GCNConv(1,1))
        self.convs.append(GINConv(mlp, eps=-1.0, train_eps=False))  # GCNConv(3, 3)
        for i in range(1, self.d - 1):
            mlp = MLP(in_channels=M*100, hidden_channels=M * 100, out_channels=M * 100, num_layers=3)
            # self.convs.append(GCNConv(1,1))
            self.convs.append(GINConv(mlp, eps=-1.0, train_eps=False))  # GCNConv(3, 3)

        mlp = MLP(in_channels=M*100, hidden_channels=M*100, out_channels=M*100, num_layers=3)
        #self.convs.append(GCNConv(1,1))
        self.convs.append(GINConv(mlp, eps=-1.0, train_eps=False))  # GCNConv(3, 3)

        self.lin1 = torch.nn.Linear(M * 100 * d, int(M * 100 * d/2))
        self.lin2 = torch.nn.Linear(int(M * 100 * d/2), C)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        xG = []
        for i in range(self.d - 1):
            x = self.convs[i](x, edge_index)
            # x = F.relu(x)
            # x = F.dropout(x, training=self.training)
            # if i == 0:
            #    n_chans = x.shape[1]
            #    running_mu = torch.zeros(n_chans) # zeros are fine for first training iter
            #    running_std = torch.ones(n_chans) # ones are fine for first training iter
            #    x = F.batch_norm(x, running_mu, running_std, training=self.training, momentum=0.9)

            xG.append(global_add_pool(x, data.batch))

        x = self.convs[self.d - 1](x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training

        xG.append(global_add_pool(x, data.batch))

        # Graph Level Readout
        xG = torch.cat(xG, dim=1)

        xG = self.lin1(xG).relu()
        xG = F.dropout(xG, training=self.training)
        xG = self.lin2(xG)  # .relu()
        return F.log_softmax(xG, dim=1)

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
        ctr += 1
        pred = model(batch.to(device)).argmax(dim=1)
        correct = (pred == batch.y).sum()
        acc += int(correct) / int(batch.y.shape[0])
    model.train()
    return acc/ctr


def main():
    # Training settings
    device = 0
    dataset_name = "graph_classification"
    batch_size = 32
    num_layer = 5
    emb_dim = 300
    drop_ratio = 0.5
    epochs = 30
    num_workers = 12
    feature = "full"
    filename = "pyq_model_gin_graph_int16.pth"

    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")

    # automatic dataloading and splitting
    dataset = GeneratedGraphClassificationDataset('/media/lorenz/Volume/code/pyq_main/pyq/app/graph/dataset/graph_classification_small/')

    split_idx = dataset.get_idx_split()


    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = GNN(d=4, M=1, C=10#C=20
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=5e-4)

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
                    torch.save({'Model' : model}, './temp_model.pth')

                test_curve.append(test_acc)
                train_curve.append(train_acc)

    bits = 16
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

    #unparser = TorchModelParser(
    #    callable_object=layer_quantizer.unwrap,
    #    callable_object_for_nonparametric=activation_quantizer.unwrap,
    #    skip_layer_by_type=skip_layer_by_type,
    #)

    model = parser.apply(torch.load("./temp_model.pth")['Model']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=5e-4)

    valid_curve = []
    test_curve = []
    train_curve = []

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

            if test_acc > best_test_acc:
                print('update')
                best_test_acc = test_acc
                torch.save({'Model' : model}, './temp_model.pth')

            test_curve.append(test_acc)
            train_curve.append(train_acc)

    if "classification" in 'classification':  # dataset.task_type:
        best_test_epoch = np.argmax(np.array(test_curve))
        best_train = max(train_curve)


    final_model = torch.load("./temp_model.pth")['Model']#best_model#model  # unparser.apply(model)
    print(eval(final_model.to(device), device, test_loader), flush=True)
    print("Finished training!")
    # print("Best validation score: {}".format(valid_curve[best_val_epoch]))
    # print("Test score: {}".format(test_curve[best_val_epoch]))

    # print(final_model._wrapped_object._wrapped_object.gnn_node.convs[0].mlp[3]._wrapped_object.weight, flush=True)
    if not filename == "":
        torch.save(
            {
                #            "Val": valid_curve[best_val_epoch],
                "Test": test_curve[best_test_epoch],
                "Train": train_curve[best_test_epoch],
                "BestTrain": best_train,
                'Model': final_model
            },
            filename,
        )


if __name__ == "__main__":
    main()
