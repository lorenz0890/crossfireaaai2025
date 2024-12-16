from pyq.fia.bfa.attack_gin_molhiv_molpcba.BFAbase import BFABase
from pyq.fia.bfa.utils import *
from torch import nn
import operator


class InjectivityBFA(BFABase):
    def __init__(self, criterion, model, k_top=10, silent=False):
        super().__init__(criterion, model, k_top, silent)


    def run_attack(self, model, data=None, target1=None, target2=None, batch_sz=None, max_flips=15, fold=0):
        data = data

        '''
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped.
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()
        # 1. perform the inference w.r.t given data and target
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        output = model(data.x, data.edge_index, data.edge_weight)
        self.loss = self.criterion(output[data.train_mask[:,fold]][target1:target1+batch_sz], output[data.train_mask[:,fold]][target2:target2+batch_sz])
        # 2. zero out the grads first, then get the grads
        for i, m in enumerate(model.modules()):
            if isinstance(m, (TransformationLayerQuantizerWrapper, GenericLayerQuantizerWrapper)):
                if not hasattr(m, 'b_w'):
                    m.N_bits = 8
                    m.b_w = nn.Parameter(2 ** torch.arange(start=m.N_bits - 1,
                                                           end=-1,
                                                           step=-1).unsqueeze(-1).float(),
                                         requires_grad=False).cuda()
                    m.b_w[0] = -m.b_w[0]  # in-place change MSB to negative
                if m._wrapped_object.weight.grad is not None:
                    m._wrapped_object.weight.grad.data.zero_()


        self.loss.backward(retain_graph=False)
        # init the loss_max to enable the while loop
        self.loss_min = self.loss.item()
        self.loss_init = self.loss.item()
        quantize(model)
        # 3. for each layer flip #bits = self.bits2flip
        #while self.loss_min >= self.loss.item():
        a=0
        for a in range(max_flips-self.bit_counter):
            print(a, self.loss_min, self.loss_init, self.loss.item(), flush=True)
            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer

            for name, module in model.named_modules():
                if isinstance(module, (TransformationLayerQuantizerWrapper, GenericLayerQuantizerWrapper)):
                    clean_weight = module._wrapped_object.weight.data.detach()  # is already quantized
                    #print(module._wrapped_object.weight.grad)
                    attack_weight = self.flip_bit(module).round().clamp(
                        module.parameter_initializer.min_threshold,
                        module.parameter_initializer.max_threshold)  # ensure range is quantized
                    # change the weight to attacked weight and get loss
                    module._wrapped_object.weight.data = attack_weight

                    dequantize(model)  # Do inference dequantized
                    output1 = model(data.x, data.edge_index, data.edge_weight)
                    #output2 = model(data.x, data.edge_index, data.edge_weight)
                    quantize(model)

                    self.loss_dict[name] = self.criterion(output1[data.train_mask[:,fold]][target1:target1+batch_sz],
                                                          output1[data.train_mask[:,fold]][target2:target2+batch_sz]).item()

                    # change the weight back to the clean weight
                    module._wrapped_object.weight.data = clean_weight

            # after going through all the layer, now we find the layer with max loss
            max_loss_module = min(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            self.loss_min = self.loss_dict[max_loss_module]
            if self.loss_min < self.loss.item():
                break
        if a == 100:
            return None

        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change that layer's weight without putting back the clean weight
        attack_log = []  # init an empty list for profile
        for module_idx, (name, module) in enumerate(model.named_modules()):
            if self.bit_counter >= max_flips: break
            if name == max_loss_module:
                attack_weight = self.flip_bit(module)

                ###########################################################
                ## Attack profiling
                #############################################
                weight_mismatch = attack_weight - module._wrapped_object.weight.detach()
                attack_weight_idx = torch.nonzero(weight_mismatch)
                if not self.silent: print('attacked module:', max_loss_module)

                attack_log = []  # init an empty list for profile
                if not self.silent: print('sz', attack_weight_idx.size(), flush=True)
                for i in range(attack_weight_idx.size()[0]):
                    weight_idx = attack_weight_idx[i, :].cpu().numpy()
                    weight_prior = module._wrapped_object.weight.detach()[
                        tuple(attack_weight_idx[i, :])].item()
                    weight_post = attack_weight[tuple(attack_weight_idx[i, :])].round().clamp(
                        module.parameter_initializer.min_threshold,
                        module.parameter_initializer.max_threshold).item()  # ensure range

                    if not self.silent:
                        print('attacked weight index:', weight_idx)
                        print('weight before attack:', weight_prior)
                        print('weight after attack:', weight_post)
                    # exit(0)
                    tmp_list = [module_idx,  # module index in the net
                                self.bit_counter + (i + 1),  # current bit-flip index
                                max_loss_module,  # current bit-flip module
                                weight_idx,  # attacked weight index in weight tensor
                                weight_prior,  # weight magnitude before attack
                                weight_post  # weight magnitude after attack
                                ]
                    attack_log.append(tmp_list)


                ###############################################################

                module._wrapped_object.weight.data = attack_weight

        # reset the bits2flip back to 0
        if not self.silent: print('Bits flipped (total, iteration):', self.bit_counter, self.n_bits2flip, flush=True)
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0
        dequantize(model)

        for i, m in enumerate(model.modules()):
            if isinstance(m, (TransformationLayerQuantizerWrapper, GenericLayerQuantizerWrapper)):
                if m._wrapped_object.weight.grad is not None:
                    m._wrapped_object.weight.grad = None

        return attack_log