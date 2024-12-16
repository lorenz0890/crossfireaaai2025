import torch
from pyq.fia.bfa.utils import *


class BFABase(object):
    def __init__(self, criterion, model, k_top=10, silent=False):

        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        self.silent = silent

        self.known_honeypots = True # Oracle Attacker
        self.msb_limit = 2 # Stealth

        # attributes for random attack
        self.module_list = []
        for name, m in model.named_modules():
            if isinstance(m, (TransformationLayerQuantizerWrapper, GenericLayerQuantizerWrapper)):
                self.module_list.append(name)

    def flip_bit(self, m):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        if self.k_top is None:
            k_top = m._wrapped_object.weight.detach().flatten().__len__()
        else:
            k_top = self.k_top

        # ADDED: 0. Oracle Attacker: zero grads of neuropots to prevent selection
        #if self.known_honeypots:
        #    m._wrapped_object.weight.grad.data[:,m.meta['neuropots_indices']] *=0

        # 1. flatten the gradient tensor to perform topk
        w_grad_topk, w_idx_topk = m._wrapped_object.weight.grad.detach().abs().view(-1).topk(k_top)

        #w_grad_topk, w_idx_topk = torch.sort(m._wrapped_object.weight.grad.detach().abs().view(-1), descending=True) #Frequency, Distribution and Ranking based staelth attacks go here.
        #w_grad_topk = w_grad_topk[k_top*2:k_top*3]
        #w_idx_topk = w_idx_topk[k_top*2:k_top*3]

        # update the b_grad to its signed representation
        w_grad_topk = m._wrapped_object.weight.grad.detach().view(-1)[w_idx_topk]

        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * m.b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() + 1) * 0.5  # zero -> negative, one -> positive

        # convert to twos complement into unsigned integer
        w_bin = int2bin(m._wrapped_object.weight.detach().view(-1), m.N_bits).short()
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(m.N_bits, 1) & m.b_w.abs().repeat(1, k_top).short()) \
                     // m.b_w.abs().repeat(1, k_top).short()
        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()

        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # ADDED: Prevent selection of 2-MSB by zeroing out positions in binary gradients
        if self.msb_limit > 0:
            b_grad_topk[:self.msb_limit,:]*=0

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip)
        bit2flip = b_grad_topk.clone().view(-1).zero_()
        #print(bit2flip.shape, b_grad_max_idx)
        #exit(-1)
        #print('GRAD MAX ITEM', grad_max.item(), flush=True)
        # print(m, flush=True)
        # exit(-1)
        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
            pass

        # 6. Based on the identified bit indexed by ```bit2flip```, generate another
        # mask, then perform the bitwise xor operation to realize the bit-flip.
        w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) \
                             ^ w_bin_topk

        # 7. update the weight in the original weight tensor
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change

        param_flipped = bin2int(w_bin,
                                m.N_bits).view(m._wrapped_object.weight.data.size()).float()

        return param_flipped

    def run_attack(self, model, data=None, target=None):
        raise NotImplementedError()