from pyq.fia.bfa.attack_gin_molhiv_molpcba.BFAbase import BFABase
from pyq.fia.bfa.utils import *
from torch import nn
import random


class RandomBFA(BFABase):
    def __init__(self, criterion, model, k_top=10, silent=False):
        super().__init__(criterion, model, k_top, silent)

    def run_attack(self, model, data=None, target=None,  max_flips=None):
        """
        Note that, the random bit-flip may not support on binary weight quantization.
        """
        for i, m in enumerate(model.modules()):
            if isinstance(m, (TransformationLayerQuantizerWrapper, GenericLayerQuantizerWrapper)):
                if not hasattr(m, 'b_w'):
                    m.N_bits = 8  # 8 # is this correct? #is_quantized --> layer has the values (not anymore?)
                    m.b_w = nn.Parameter(2 ** torch.arange(start=m.N_bits - 1,
                                                           end=-1,
                                                           step=-1).unsqueeze(-1).float(),
                                         requires_grad=False).cuda()
                if m._wrapped_object.weight.grad is not None:
                    m._wrapped_object.weight.grad.data.zero_()
        # exit(-1)
        quantize(model)
        chosen_module = random.choice(self.module_list)
        for module_idx, (name, m) in enumerate(model.named_modules()):
            if name == chosen_module:
                flatten_weight = m._wrapped_object.weight.detach().view(-1)
                chosen_idx = random.choice(range(flatten_weight.__len__()))
                # convert the chosen weight to 2's complement
                bin_w = int2bin(flatten_weight[chosen_idx], m.N_bits).short()
                # randomly select one bit
                bit_idx = random.choice(range(m.N_bits))
                mask = (bin_w.clone().zero_() + 1) * (2 ** bit_idx)
                bin_w = bin_w ^ mask
                int_w = bin2int(bin_w, m.N_bits).float()

                ##############################################
                ###   attack profiling
                ###############################################

                weight_mismatch = flatten_weight[chosen_idx] - int_w
                attack_weight_idx = chosen_idx

                if not self.silent: print('attacked module:', chosen_module)

                attack_log = []  # init an empty list for profile

                weight_idx = chosen_idx
                weight_prior = flatten_weight[chosen_idx]
                weight_post = int_w

                if not self.silent:
                    print('attacked weight index:', weight_idx)
                    print('weight before attack:', weight_prior)
                    print('weight after attack:', weight_post)

                tmp_list = ["module_idx",  # module index in the net
                            self.bit_counter + 1,  # current bit-flip index
                            chosen_module,  # "loss",  # current bit-flip module
                            [weight_idx // m._wrapped_object.weight.data.shape[1],
                             weight_idx % m._wrapped_object.weight.data.shape[1]],
                            # attacked weight index in weight tensor
                            weight_prior,  # weight magnitude before attack
                            weight_post  # weight magnitude after attack
                            ]
                attack_log.append(tmp_list)

                self.bit_counter += 1
                #################################

                flatten_weight[chosen_idx] = int_w
                m._wrapped_object.weight.data = flatten_weight.view(
                    m._wrapped_object.weight.data.size())
        dequantize(model)

        return attack_log