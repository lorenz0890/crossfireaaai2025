import hashlib
import sys
import traceback
from collections import defaultdict

import networkx as nx
import torch.nn
from matplotlib import pyplot as plt

from torch.nn.utils.parametrizations import orthogonal
from torch_geometric.utils import to_dense_adj
from torchvision.utils import make_grid

from app.graph.gin_molhiv_moltox_quantization import *
from ogb.graphproppred import PygGraphPropPredDataset as ogb_datasets
from torch_geometric.data import DataLoader as GMDataLoader

from pyq import GenericLayerQuantizerWrapper
from pyq.fia.bfa.attack_gin_molhiv_molpcba.RBFA import RandomBFA as RBFA
from pyq.fia.bfa.attack_gin_molhiv_molpcba.PBFA import ProgressiveBFA as PBFA
from pyq.fia.bfa.attack_gin_molhiv_molpcba.IBFA import InjectivityBFA as IBFA
import datetime
import time
import random
from ogb.graphproppred import Evaluator

from pyq.fia.bfa.utils import _WrappedGraphDataset, eval_ogbg, get_n_params, dequantize, quantize
from sklearn.metrics import roc_auc_score
import argparse #new
import json

import numpy as np
import scipy.stats as stat
import torch.nn.utils.prune as prune

from scipy.optimize import minimize
from scipy.optimize import dual_annealing

from scipy.stats import iqr

from torch_geometric.datasets.fake import FakeDataset
from torch_geometric.loader.dataloader import DataLoader

def setup_data(dataset_name, batch_size, shuffle=True, ibfa_limit=0.01):
    dataset = ogb_datasets(name=dataset_name)
    split_idx = dataset.get_idx_split()
    #train_data = dataset[split_idx["train"]]
    #valid_data = dataset[split_idx["valid"]]
    test_data = dataset[split_idx["test"]]
    perm = torch.randperm(int(split_idx["train"].size(0)))
    idx = perm[:max(int(split_idx["train"].shape[0]*ibfa_limit), 2*batch_size)]
    samples = split_idx["train"][idx]
    train_data = dataset[samples]

    train_loader = GMDataLoader(dataset=_WrappedGraphDataset(train_data, None),
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=12, pin_memory=True)
    test_loader = GMDataLoader(dataset=_WrappedGraphDataset(test_data, None),
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=12, pin_memory=True)
    return train_loader, test_loader

def setup_net(dataset_name):
    if dataset_name == 'ogbg-molpcba':
        net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/pyq_model_gin_molpcba.pth")['Model']
    elif dataset_name == 'ogbg-molhiv':
        net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/pyq_model_gin_molhiv.pth")['Model']
    elif dataset_name == 'ogbg-moltox21':
        net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/pyq_model_gin_moltox.pth")['Model']
    elif dataset_name == 'ogbg-molclintox':
        net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/pyq_model_gin_molclintox.pth")['Model']
    elif dataset_name == 'ogbg-moltoxcast':
        net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/pyq_model_gin_moltoxcast.pth")['Model']
    elif dataset_name == 'ogbg-molbace':
        net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/pyq_model_gin_molbace.pth")['Model']
    elif dataset_name == 'ogbg-molbbbp':
        net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/pyq_model_gin_molbbbp.pth")['Model']
    elif dataset_name == 'ogbg-molsider':
        net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/pyq_model_gin_molsider.pth")['Model']
    elif dataset_name == 'ogbg-molmuv':
        net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/pyq_model_gin_molmuv.pth")['Model']
    net.eval()
    return net

def setup_attack(attack_type, dataset):
    BFA, criterion, data, target = None, None, None, None
    if attack_type == 'RBFA':
        BFA = RBFA
        criterion = torch.nn.BCEWithLogitsLoss()
    if attack_type == 'PBFA':
        BFA = PBFA
        criterion = torch.nn.BCEWithLogitsLoss()
    if attack_type in ['IBFAv1', 'IBFAv2']:
        BFA = IBFA
        if dataset in ['ogbg-molpcba']:
            criterion = torch.nn.KLDivLoss(log_target=True)
        if dataset in ['ogbg-moltox21', 'ogbg-moltoxcast', 'ogbg-molsider', 'ogbg-molclintox', 'ogbg-molmuv']:
            criterion = torch.nn.KLDivLoss(log_target=True)
        #L1 doesnt work well for molpcba
        if dataset in ['ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp']:
            criterion = torch.nn.L1Loss()# L1 works better for molhiv than KLDDiv
    return BFA, criterion

def tensor_to_string(self):
    return str(self.tolist())
setattr(torch.Tensor, 'tostring', tensor_to_string)
#https://discuss.pytorch.org/t/i-wrote-some-code-to-add-a-tostring-function-to-tensors/93957

def main():
    parser = argparse.ArgumentParser(prog='python run_bfa_gin_molecular.py')
    parser.add_argument('--type', help='PBFA, RBFA, IBFAv1, IBFAv2', type=str)
    parser.add_argument('--data', help='ogbg-molpcba, ogbg-molhiv, ogbg-moltox21, ogbg-molclintox, ogbg-moltoxcast, ogbg-molbace', type=str)
    parser.add_argument('--n', help='Run experiment n times', type=int)
    parser.add_argument('--k', help='Run BFA k times', type=int)
    parser.add_argument('--npc', help='Neuropots percentage', type=float)
    parser.add_argument('--npg', help='Neuropots gamma', type=float)
    parser.add_argument('--sz', help='Batch size', type=int)
    parser.add_argument('--lim', help='IBFA data usage limit', type=float)
    args = parser.parse_args()
    print(args.type, args.data, args.n, args.k, args.sz, args.npc, args.npg)

    attack_type, dataset_name, device = args.type, args.data, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_runs, eval_runs, attack_runs, batch_size = args.n, 10, args.k, args.sz

    ibfa_lim = args.lim
    if not 'IBFA' in attack_type:
        ibfa_lim = 1.0

    pre_metric, mod_metric, post_metric, bit_flips, pre_repair_metric, detection_rate, reconstr_rate = [], [], [], [], [], [], []
    attack_cost = []

    print('Start time', datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    experiment_accumulated_seconds = 0
    failure_counter = 0
    evaluator = Evaluator(dataset_name)

    for r in range(experiment_runs):
        start_time = time.time()
        train_loader, test_loader = setup_data(dataset_name, batch_size, False, ibfa_lim)  # shuffled every epoch -> should be False
        net = setup_net(dataset_name).to(device)#copy.deepcopy(net_clean).to(device)
        BFA, criterion = setup_attack(attack_type, dataset_name)
        print('params', get_n_params(net))
        data, target = None, None
        if attack_type == 'PBFA':
            with torch.no_grad():
                x = random.randint(0, int(len(train_loader.dataset)/batch_size))
                for i, data in enumerate(train_loader):
                    if i == x:
                        print('PBFA data found', flush=True)
                        data = data.to(device)
                        target = torch.round(torch.nn.functional.sigmoid(net(data)))
                        #target = data.y
                        break
        if attack_type == 'RBFA':
            pass
        if attack_type == 'IBFAv1':
            with torch.no_grad():
                max_loss = 0
                d = train_loader#d = [x for x in train_loader]
                for i, data1 in enumerate(d):
                    for j, data2 in enumerate(d):
                        if True: #data1.size() == data2.size() and i != j:
                            data1 = data1.to(device)
                            data2 = data2.to(device)

                            is_labeled1, is_labeled2 = data1.y == data1.y, data2.y == data2.y # Not all data always labeled.
                            out1, out2 = net(data1)[is_labeled1], net(data2)[is_labeled2]
                            cut = min(out1.shape[0], out2.shape[0])
                            loss = criterion(out1[:cut], out2[:cut])
                            if max_loss < loss:
                                print(i,j, loss)
                                max_loss = loss
                                max_data1 = data1
                                max_data2 = data2
                        #if j > 10: break
                    #if i > 1:
                    #    break
            print(max_data1.size(), max_data2.size(), flush=True)
            print('IBFAv1 data found', flush=True)
            data = max_data1
            target = max_data2

        net = net.to(device)

        test_perf = eval_ogbg(net, device, test_loader, evaluator)
        pre_metric.append(list(test_perf.values())[0])

        # Prune to artificially increase attack surface and induce "natural" neuropots (which are later selected via gradients)
        for name, module in net.named_modules():
            if hasattr(module, '_wrapped_object') and  hasattr(module._wrapped_object, 'weight'):
                pruning_ratio = 0.25 # Actually the ratio of remaining weights
                num_weights_to_prune = int((1-pruning_ratio) * module._wrapped_object.weight.data.numel())
                abs_weights = module._wrapped_object.weight.data.abs()
                sorted_abs_weights, sorted_indices = torch.sort(abs_weights.view(-1), descending=True)
                threshold_weight = sorted_abs_weights[num_weights_to_prune]
                pruning_mask = (abs_weights >= threshold_weight).float()
                module._wrapped_object.weight.data = module._wrapped_object.weight.data * pruning_mask

        # Generate grads before making hook
        grads = {}
        for name, module in net.named_modules():
            if hasattr(module, '_wrapped_object') and hasattr(module._wrapped_object, 'weight'):
                grads[name] = torch.zeros_like(module._wrapped_object.weight)

        # Randomly sample different data points
        indices = torch.randperm(len(train_loader.dataset))[:10]  # Sample 10 indices
        for ind1 in indices:
            print('Defensive gradient ranking', flush=True)

            net.zero_grad()

            data1 = train_loader.dataset[ind1].to(device)
            out1 = net(data1)

            def_target1 = torch.round(torch.nn.functional.sigmoid(out1)).detach()

            selection_criterion = torch.nn.BCEWithLogitsLoss()
            loss1 = selection_criterion(out1.float(), def_target1.float())

            loss1.backward(create_graph=True)

            for name, module in net.named_modules():
                if name in grads:
                    if not torch.isnan(module._wrapped_object.weight.grad).any().item():
                        grads[name] += module._wrapped_object.weight.grad

        for name, module in net.named_modules():
            if name in grads:
                module._wrapped_object.grad = grads[name].detach()

        print('Ranking done')

        #Securely store meta information of clean net
        handles = []
        gamma = args.npg
        neuropots_perc = args.npc

        depth = [1]

        def compute_checksum(matrix, axis=0):
            matrix = matrix.to(torch.int8)
            if axis == 0:
                chunks = torch.split(matrix, 1, dim=1)
            elif axis == 1:
                chunks = torch.split(matrix, 1, dim=0)
            else:
                raise ValueError("Invalid axis. Must be 0 (rows) or 1 (columns).")
            hashes = torch.tensor([int.from_bytes(hashlib.blake2b(chunk.squeeze().char().cpu().numpy().tobytes(), digest_size=2).digest(), 'big') for chunk in chunks], dtype=torch.int16) # 2 bytes
            #hashes = torch.tensor([int.from_bytes(hashlib.blake2b(chunk.squeeze().char().cpu().numpy().tobytes(), digest_size=1).digest(), 'big') for chunk in chunks], dtype=torch.uint8) #1 byte
            return hashes

        module_handles = []
        def one_shot_crossfire(name, setup):
            def hook(module, input):
                num_neurons = module._wrapped_object.weight.data.shape[1]
                if setup and not hasattr(module, 'meta'): #first iteration
                    module.module_handle_index = len(module_handles)
                    module_handles.append(module)

                    #Ranked Neuropots
                    abs_grads = torch.abs(module._wrapped_object.weight.grad.data)
                    depth[0] *=1.1

                    num_pots = max(int(num_neurons * neuropots_perc), 2) # at least 2 pots

                    values, indices = torch.sort(torch.sum(abs_grads, dim=0), descending=True)
                    ind = indices[:num_pots]
                    scale = gamma
                    scale *=depth[0]
                    neuropot_indices = ind.cuda()

                    scores = (values)[:num_pots]
                    min_score = torch.min(scores)
                    max_score = torch.max(scores)
                    gamma_hat = 1 + (scores - min_score) * (scale - 1) / (max_score - min_score)

                    module._wrapped_object.weight.data[:, neuropot_indices] *= 1 / gamma_hat

                    quantize(module)

                    module.min = module._wrapped_object.weight.data.min() #min
                    module.max = module._wrapped_object.weight.data.max() #max


                    module.meta = {
                                  'seed' : 123,
                                  'neuropots' : module._wrapped_object.weight.data[:, neuropot_indices].detach(),
                                  'neuropots_indices' : neuropot_indices,
                                  'integrity_hash' : hashlib.blake2b(module._wrapped_object.weight.data.char().cpu().numpy().tobytes(),digest_size=2).hexdigest(), # Alternative digest: 4
                                  'gamma' : gamma_hat,
                                  }

                    module.meta["chk0"] = compute_checksum(module._wrapped_object.weight.data.detach().cpu().clone(), axis=0)
                    module.meta["chk1"] = compute_checksum(module._wrapped_object.weight.data.detach().cpu().clone(), axis=1)


                    dequantize(module)

                if hasattr(module, 'meta') and 'neuropots_indices' in module.meta:
                    input[:, module.meta['neuropots_indices']] *= module.meta['gamma']

                return input
            return hook
        d = [0]
        def hook_all_layers(net, d, set_hook_2, store):
            for name, layer in net._modules.items():
                if isinstance(layer, (
                        GNN,
                        GNN_node,
                        GNN_node,
                        GINConv,
                        torch.nn.modules.container.ModuleList,
                )):
                    hook_all_layers(layer, d, set_hook_2, store)
                else:
                    if 'mlp' in name or 'linear' in name:
                        if isinstance(layer, TransformationLayerQuantizerWrapper):
                            handles.append(layer.register_forward_pre_hook(set_hook_2("graph_pred_linear", store))) #TODO fix this in the other approaches too
                            #handles.append(layer.register_backward_hook(set_hook_2("graph_pred_linear", store)))  # TODO fix this in the other approaches too
                        for subname, submodule in layer._modules.items():
                            if isinstance(submodule, TransformationLayerQuantizerWrapper):
                                handles.append(submodule.register_forward_pre_hook(set_hook_2("convs.{}.{}.{}".format(d[0], name, subname), store)))
                        d[0] += 1

        hook_all_layers(net, d, one_shot_crossfire, True)
        test_perf_mod = eval_ogbg(net, device, test_loader, evaluator)
        print(test_perf, test_perf_mod)

        for handle in handles: handle.remove()
        hook_all_layers(net, d, one_shot_crossfire, False)

        # Run attack
        logs = []
        attacker, attack_log = BFA(criterion, net, 100, True), None
        flips=0
        try:
            if attack_type in ['RBFA', 'PBFA', 'IBFAv1']:
                for i in range(attack_runs):
                    attack_start_time = time.time()
                    attack_log = attacker.run_attack(net, data, target, attack_runs)
                    attack_cost.append(time.time() - attack_start_time)
                    if len(attack_log) > 0:
                        flips = attack_log[-1][1]
                    else:
                        break
                    logs.append(attack_log)
            if attack_type == 'IBFAv2':
                for _ in range(attack_runs):
                    with torch.no_grad():
                        max_loss = 0
                        for i, data1 in enumerate(train_loader):
                            for j, data2 in enumerate(train_loader):
                                if True: #data1.size() == data2.size() and i != j:
                                    data1 = data1.to(device)
                                    data2 = data2.to(device)

                                    is_labeled1, is_labeled2 = data1.y == data1.y, data2.y == data2.y  # Not all data always labeled.
                                    out1, out2 = net(data1)[is_labeled1], net(data2)[is_labeled2]
                                    cut = min(out1.shape[0], out2.shape[0])
                                    loss = criterion(out1[:cut], out2[:cut])

                                    if max_loss < loss:
                                        max_loss = loss
                                        max_data1 = data1
                                        max_data2 = data2
                            #if i > 5: # For quick test, limit search space
                            #    break
                    attack_log = attacker.run_attack(net, max_data1.to(device), max_data2.to(device), attack_runs)
                    if len(attack_log) > 0:
                        flips = attack_log[-1][1]
                    else:
                        break
                    logs.append(attack_log)
            if attack_log is None:
                raise Exception('No attack solution found')

            bit_flips.append(flips)
            pre_repair = eval_ogbg(net, device, test_loader, evaluator)
            pre_repair_metric.append(list(pre_repair.values())[0])

            # Attempt repair
            d = [0]
            flips_detected = [0]
            integrity=[True]

            def detect_repair_crossfire(net, d, flips_detected):
                for name, layer in net._modules.items():
                    if isinstance(layer, (GNN, GNN_node, GINConv, torch.nn.modules.container.ModuleList)):
                        detect_repair_crossfire(layer, d, flips_detected)
                    else:
                        if 'mlp' in name or 'linear' in name:
                            if isinstance(layer, TransformationLayerQuantizerWrapper):
                                flip_list_1, flip_list_0 = [], []
                                subname = 'N/A'
                                quantize(layer)
                                chk0 = compute_checksum(layer._wrapped_object.weight.data.detach().cpu().clone(), axis=0)
                                chk1 = compute_checksum(layer._wrapped_object.weight.data.detach().cpu().clone(), axis=1)
                                if layer.meta['integrity_hash'] == hashlib.blake2b(layer._wrapped_object.weight.data.char().cpu().numpy().tobytes(), digest_size=2).hexdigest():
                                    dequantize(layer)
                                    continue

                                use_range_ok = layer.min.item() - layer._wrapped_object.weight.data.min().item() + layer.max.item() - layer._wrapped_object.weight.data.max().item()
                                if use_range_ok != 0:
                                    indices = torch.nonzero(torch.logical_or(layer._wrapped_object.weight.data < layer.min, layer._wrapped_object.weight.data > layer.max))
                                    for idx in indices:
                                        i, j = idx
                                        e = layer._wrapped_object.weight.data[i, j].detach().clone()
                                        int_val = e.to(torch.int8)
                                        if j in layer.meta['neuropots_indices']:
                                            potj = torch.where(j == layer.meta['neuropots_indices'])
                                            layer._wrapped_object.weight.data[i, j] = layer.meta['neuropots'][i, potj]
                                        elif int_val == -128:
                                            layer._wrapped_object.weight.data[i, j] = 0
                                        else:
                                            while not (layer.min <= int_val <= layer.max):
                                                for k in range(7, -1, -1):
                                                    if int_val & (1 << k):  # If the bit is set
                                                        int_val &= ~(1 << k)  # Unset the bit
                                                        break
                                            layer._wrapped_object.weight.data[i, j] = float(int_val)

                                        for log in logs:
                                            for attack in log:
                                                if "graph_pred_linear" in attack[2] or "convs.{}.{}.{}".format(d[0], name, subname) in  attack[2]:
                                                    flips_detected[0] += int(attack[3][1] == j and attack[3][0] == i)  # Mitigation rate
                                                    flip_list_1.append(j)  # Prevent double accounting
                                                    flip_list_0.append(i)

                                if hasattr(layer, 'meta'):  # First iteration
                                    bool0 = chk0 != layer.meta['chk0']
                                    bool1 = chk1 != layer.meta['chk1']
                                    if bool0.any().item() and bool1.any().item():
                                        dim0 = torch.nonzero(bool0).flatten()
                                        dim1 = torch.nonzero(bool1).flatten()
                                        if dim0.shape[0] > 0 and dim1.shape[0] > 0:
                                            for idx0 in dim0:
                                                for idx1 in dim1:
                                                    if idx0 in layer.meta['neuropots_indices']:
                                                        pot0 = torch.where(idx0 == layer.meta['neuropots_indices'])
                                                        layer._wrapped_object.weight.data[idx1, idx0] = layer.meta['neuropots'][idx1,pot0]
                                                    elif idx0 not in flip_list_1 and idx1 not in flip_list_0:
                                                        layer._wrapped_object.weight.data[idx1, idx0] = 0
                                                    for log in logs:
                                                        for attack in log:
                                                            if "graph_pred_linear" in attack[2] or "convs.{}.{}.{}".format(d[0], name, subname) in attack[2]:
                                                                if idx0 not in flip_list_1 and idx1 not in flip_list_0:  # Prevent double accounting
                                                                    flips_detected[0] += int(attack[3][1] == idx0 and attack[3][0] == idx1)  # Mitigation rate
                                            layer.meta['chk0'] = compute_checksum(layer._wrapped_object.weight.data.detach().cpu().clone(), axis=0)
                                            layer.meta['chk1'] = compute_checksum(layer._wrapped_object.weight.data.detach().cpu().clone(), axis=1)
                                integrity[0] = integrity[0] and layer.meta['integrity_hash'] == hashlib.blake2b(layer._wrapped_object.weight.data.char().cpu().numpy().tobytes(), digest_size=2).hexdigest()
                                dequantize(layer)

                            # Repeat for all submodule found
                            for subname, submodule in layer._modules.items():
                                if isinstance(submodule, TransformationLayerQuantizerWrapper):
                                    flip_list_1, flip_list_0 = [], []
                                    quantize(submodule)
                                    chk0 = compute_checksum(submodule._wrapped_object.weight.data.detach().cpu().clone(), axis=0)
                                    chk1 = compute_checksum(submodule._wrapped_object.weight.data.detach().cpu().clone(), axis=1)
                                    if submodule.meta['integrity_hash'] == hashlib.blake2b(submodule._wrapped_object.weight.data.char().cpu().numpy().tobytes(), digest_size=2).hexdigest():
                                        dequantize(submodule)
                                        continue

                                    use_range_ok = submodule.min.item() - submodule._wrapped_object.weight.data.min().item() + submodule.max.item() - submodule._wrapped_object.weight.data.max().item()
                                    if use_range_ok != 0:
                                        indices = torch.nonzero(torch.logical_or(submodule._wrapped_object.weight.data < submodule.min, submodule._wrapped_object.weight.data > submodule.max))
                                        for idx in indices:
                                            i, j = idx
                                            e = submodule._wrapped_object.weight.data[i, j].detach().clone()
                                            int_val = e.to(torch.int8)
                                            if j in submodule.meta['neuropots_indices']:
                                                potj = torch.where(j == submodule.meta['neuropots_indices'])
                                                submodule._wrapped_object.weight.data[i, j] = submodule.meta['neuropots'][i, potj]
                                            elif int_val == -128:
                                                submodule._wrapped_object.weight.data[i, j] = 0
                                            else:
                                                while not (submodule.min <= int_val <= submodule.max):
                                                    for k in range(7, -1, -1):
                                                        if int_val & (1 << k):  # If the bit is set
                                                            int_val &= ~(1 << k)  # Unset the bit
                                                            break

                                                submodule._wrapped_object.weight.data[i, j] = float(int_val)

                                            for log in logs:
                                                for attack in log:
                                                    if "graph_pred_linear" in attack[2] or "convs.{}.{}.{}".format(d[0], name, subname) in attack[2]:
                                                        flips_detected[0] += int(attack[3][1] == j and attack[3][0] == i)  # Mitigation rate
                                                        flip_list_1.append(j) #Prevent double accounting
                                                        flip_list_0.append(i)

                                    if hasattr(submodule, 'meta'):  # first iteration
                                        bool0 = chk0 != submodule.meta['chk0']
                                        bool1 = chk1 != submodule.meta['chk1']
                                        if bool0.any().item() and bool1.any().item():
                                            dim0 = torch.nonzero(bool0).flatten()
                                            dim1 = torch.nonzero(bool1).flatten()
                                            if dim0.shape[0] > 0 and dim1.shape[0] > 0:
                                                for idx0 in dim0:
                                                    for idx1 in dim1:
                                                        if idx0 in submodule.meta['neuropots_indices']:
                                                            pot0 = torch.where(idx0 == submodule.meta['neuropots_indices'])
                                                            submodule._wrapped_object.weight.data[idx1, idx0] = submodule.meta['neuropots'][idx1, pot0]
                                                        elif idx0 not in flip_list_1 and idx1 not in flip_list_0:
                                                            submodule._wrapped_object.weight.data[idx1, idx0] = 0
                                                        for log in logs:
                                                            for attack in log:
                                                                if "graph_pred_linear" in attack[2] or "convs.{}.{}.{}".format(d[0],name,subname) in attack[2]:
                                                                    if idx0 not in flip_list_1  and idx1 not in flip_list_0: #Prevent double accounting
                                                                        flips_detected[0] += int(attack[3][1] == idx0 and attack[3][0] == idx1)  # Mitigation rate
                                                submodule.meta['chk0'] = compute_checksum(submodule._wrapped_object.weight.data.detach().cpu().clone(), axis=0)
                                                submodule.meta['chk1'] = compute_checksum(submodule._wrapped_object.weight.data.detach().cpu().clone(), axis=1)
                                    # Check integrity - maybe the zeros in the groups did the trick
                                    integrity[0] = integrity[0] and submodule.meta['integrity_hash'] == hashlib.blake2b(submodule._wrapped_object.weight.data.char().cpu().numpy().tobytes(),digest_size=2).hexdigest()
                                    dequantize(submodule)
                            d[0] += 1

            #for log in logs:
            #    for attack in log:
            #        print(attack)
            print('Crossfire Repair')
            integrity[0] = True
            d = [0]
            detect_repair_crossfire(net, d, flips_detected)
            detection_rate.append(flips_detected[0]/sum(bit_flips))#[-1]
            reconstr_rate.append(int(integrity[0]))
            test_perf = eval_ogbg(net, device, test_loader, evaluator)
            print('Mod Net', test_perf_mod, 'Pre Repair:', pre_repair, 'Post Repair:', test_perf)
            mod_metric.append(list(test_perf_mod.values())[0])
            post_metric.append(list(test_perf.values())[0])

        except Exception as e:
            failure_counter+=1
            print(failure_counter, e.args)
            print(traceback.format_exc())
        ct = datetime.datetime.now()
        experiment_accumulated_seconds += time.time() - start_time

        print('Current time:', ct.strftime("%d/%m/%Y, %H:%M:%S"),
              'Completed:', (r + 1) / experiment_runs * 100, '%',
              'Duration per experiment:', round(time.time() - start_time, 2), 's',
              'ETA:', (ct+datetime.timedelta(seconds=((experiment_accumulated_seconds/(r+1)) * (experiment_runs-r-1)))).strftime("%d/%m/%Y, %H:%M:%S")
        )
    # TODO: Detection rate recording faulty for multiple runs. Output needs to be divided by (1/k) * sum_i=1^k(1/i) and multiplied by (1-n)
    #  to correct with k = number of runs and n = number of failed runs (runs where exception was raised)
    # TODO: Recovery rate recording faulty. Output needs to be multiplied by (1-n) where n =  failed runs.
    print("Clean GNN {} ({}), Mod GNN {} ({}) , Post-BFA GNN {} ({}), Repaired GNN {} ({}), Flips {} ({}), Detection rate {} ({}), Recovery rate {} ({}), Attack cost {} ({}), Failures {}".format(
        #'Pre attack,  and post attack average metric, bitflips, flips detected, full recovery, failures',
        round(np.array(pre_metric).mean(), 2), round(np.array(pre_metric).std(), 2),
        round(np.array(mod_metric).mean(), 2), round(np.array(mod_metric).std(), 2),
        round(np.array(pre_repair_metric).mean(), 2), round(np.array(pre_repair_metric).std(), 2),
        round(np.array(post_metric).mean(), 2), round(np.array(post_metric).std(), 2),
        round(np.array(bit_flips).mean(), 2), round(np.array(bit_flips).std(), 2),
        round(np.array(detection_rate).mean(), 2), round(np.array(detection_rate).std(), 2),
        round(np.array(reconstr_rate).mean(), 2), round(np.array(reconstr_rate).std(), 2),
        round(np.array(attack_cost).mean(), 2), round(np.array(attack_cost).std(), 2),
        failure_counter)
        )
    print('Start time', datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))


    json_string = json.dumps({
        'clean' : [round(np.array(pre_metric).mean(), 2), round(np.array(pre_metric).std(), 2)],
        'mod' : [round(np.array(mod_metric).mean(), 2), round(np.array(mod_metric).std(), 2)],
        'pre' : [round(np.array(pre_repair_metric).mean(), 2), round(np.array(pre_repair_metric).std(), 2)],
        'post' : [round(np.array(post_metric).mean(), 2), round(np.array(post_metric).std(), 2)],
        'flips' : [round(np.array(bit_flips).mean(), 2), round(np.array(bit_flips).std(), 2)],
        'det' : [round(np.array(detection_rate).mean(), 2), round(np.array(detection_rate).std(), 2)],
        'recr' : [round(np.array(reconstr_rate).mean(), 2), round(np.array(reconstr_rate).std(), 2)],
        'atk' : [round(np.array(attack_cost).mean(), 2), round(np.array(attack_cost).std(), 2)],
        'fails' : failure_counter,
        'gamma': args.npg,
        'perc': args.npc,
     })

    with open("CROSSFIRE_{}_{}_{}_{}_{}_{}_{}_{}.json".format(attack_type, dataset_name, device, experiment_runs, attack_runs, batch_size, str(args.npg).replace('.', ''), str(args.npc).replace('.', '')), 'w') as outfile:
        outfile.write(json_string)

if __name__ == "__main__":
    main()
