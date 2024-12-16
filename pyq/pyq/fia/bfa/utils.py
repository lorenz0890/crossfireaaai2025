import torch
from torchmetrics import AUROC

#from models.quantization import quan_Conv2d, quan_Linear
from pyq.core.quantization.wrapper import GenericLayerQuantizerWrapper, TransformationLayerQuantizerWrapper
from tqdm import tqdm
class _WrappedGraphDataset:
    """Allows to add transforms to a given Dataset."""

    def __init__(self,
                 dataset,
                 transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]  # sample, label


def evaluate_molhiv(net, data_loader, device, runs=10):
    auroc_list = []
    auroc = AUROC(pos_label=1)
    net.eval()
    with torch.no_grad():
        for i in range(runs):
            for _, data in enumerate(data_loader):
                data = data.to(device)  # .cuda()
                target = data.y
                out = net(data)
                auroc_list.append(auroc(out.t()[0], target.t()[0]))
    return sum(auroc_list)/len(auroc_list)


def eval_ogbg(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)

def evaluate_cora(net, data_loader, device, runs=10):
    acc_list = []
    net.eval()
    with torch.no_grad():
        for i in range(runs):
            for _, data in enumerate(data_loader):
                data = data.to(device)
                pred = net(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)
                acc_list.append(int((pred[data.val_mask] == data.y[data.val_mask]).sum()) / int(data.val_mask.sum()))
    return sum(acc_list)/len(acc_list)

def get_n_params(model):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/5
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def quantize(model):
    for i, m in enumerate(model.modules()):
        if isinstance(m, TransformationLayerQuantizerWrapper):
            m._wrapped_object.weight.data = m._wrapped_object.weight.data / m.parameter_initializer.scale
            m._wrapped_object.weight.data = m._wrapped_object.weight.data.round().clamp(
                m.parameter_initializer.min_threshold,
                m.parameter_initializer.max_threshold)

            if hasattr(m._wrapped_object.bias, "data"):
                m._wrapped_object.bias.data = m._wrapped_object.bias.data / m.parameter_initializer.scale
                m._wrapped_object.bias.data = m._wrapped_object.bias.data.round().clamp(
                    m.parameter_initializer.min_threshold,
                    m.parameter_initializer.max_threshold)


def dequantize(model):
    # Dequantize for evaluation
    for i, m in enumerate(model.modules()):
        if isinstance(m, TransformationLayerQuantizerWrapper):
            m._wrapped_object.weight.data = m._wrapped_object.weight.data * m.parameter_initializer.scale
            if hasattr(m._wrapped_object.bias, "data"):
                m._wrapped_object.bias.data = m._wrapped_object.bias.data * m.parameter_initializer.scale

def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    Note that, the conversion is different depends on number of bit used.
    '''
    output = input.clone()
    if num_bits == 1: # when it is binary, the conversion is different
        output = output/2 + .5
    elif num_bits > 1:
        output[input.lt(0)] = 2**num_bits + output[input.lt(0)]

    return output


def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    if num_bits == 1:
        output = input*2-1
    elif num_bits > 1:
        mask = 2**(num_bits - 1) - 1
        output = -(input & ~mask) + (input & mask)
    return output


def weight_conversion(model):
    '''
    Perform the weight data type conversion between:
        signed integer <==> two's complement (unsigned integer)
    Such conversion is used as additional step to ensure the conversion correctness

    Note that, the data type conversion chosen is depend on the bits:
        N_bits <= 8   .char()   --> torch.CharTensor(), 8-bit signed integer
        N_bits <= 16  .short()  --> torch.shortTensor(), 16 bit signed integer
        N_bits <= 32  .int()    --> torch.IntTensor(), 32 bit signed integer
    '''
    for m in model.modules():
        if isinstance(m, TransformationLayerQuantizerWrapper):
            w_bin = int2bin(m._wrapped_object.weight.data, m.N_bits).short()
            m._wrapped_object.weight.data = bin2int(w_bin, m.N_bits).float()
    return


def count_ones(t, n_bits):
    counter = 0
    for i in range(n_bits):
        counter += ((t & 2**i) // 2**i).sum()
    return counter.item()


def hamming_distance(model1, model2):
    '''
    Given two model whose structure, name and so on are identical.
    The only difference between the model1 and model2 are the weight.
    The function compute the hamming distance bewtween the bianry weights
    (two's complement) of model1 and model2.
    '''
    # TODO: add the function check model1 and model2 are same structure
    # check the keys of state_dict match or not.

    H_dist = 0  # hamming distance counter

    #quantize(model1)
    #quantize(model2)
    for name, module in model1.named_modules():
        if isinstance(module, GenericLayerQuantizerWrapper):
            # remember to convert the tensor into integer for bitwise operations
            #print(name, flush=True)
            #print(model1.state_dict().__dict__, flush=True)
            binW_model1 = int2bin(model1.state_dict()[name + '._wrapped_object.lin._wrapped_object.weight'],
                                  module.N_bits).short()
            binW_model2 = int2bin(model2.state_dict()[name + '._wrapped_object.lin._wrapped_object.weight'],
                                  module.N_bits).short()
            H_dist += count_ones(binW_model1 ^ binW_model2, module.N_bits)

    #dequantize(model1)
    #dequantize(model2)
    return H_dist