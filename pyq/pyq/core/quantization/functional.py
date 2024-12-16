from math import sqrt

from torch import bernoulli, bool, ge, sum
from torch.autograd import Function
from torch.nn import Module

DEFAULT_CTX_NAME = "ctx"
DEFAULT_FUNCTION_ARGUMENTS_DICT_NAME = "get_function_arguments_dict"

# use .floor() instead of .round() for symmetric quantization

"""
import matplotlib.pyplot as plt

# the histogram of the data
n, bins, patches = plt.hist(x.reshape(-1).cpu().detach().numpy().tolist(), 50, density=True, facecolor='g', alpha=0.75)

plt.grid(True)
plt.show()
"""


# TODO (Samir): Round the z offset parameter


def bernoulli_probability_to_mask(probability):
    return bernoulli(probability).to(bool)


class ModuleFunction(Module):
    def __call__(ctx, *args, **kwargs):
        return ctx.forward(*args, **kwargs)

    def forward(ctx, *args, **kwargs):
        raise NotImplementedError("{} forward is not implemented.".format(ctx.__class__.__name__))

    def apply(ctx, *args, **kwargs):
        return ctx.forward(*args, **kwargs)


class PyQFunction(Function):
    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()
        # stored in the pickle
        return state

    def __setstate__(self, newstate):
        # re-instate our __dict__ state from the pickled state
        self.__dict__.update(newstate)

    def __repr__(self):
        return self.__class__.__name__


class NoneOffsetFunction(PyQFunction):
    pass


class OffsetFunction(PyQFunction):
    pass


class IdentityFunction(PyQFunction):
    @staticmethod
    def forward(ctx, x):
        x = x.round()
        return x, x

    @staticmethod
    def backward(ctx, grad_hat_one_output, grad_hat_two_output):
        return grad_hat_one_output.clone()


class DeQuantizer(NoneOffsetFunction):
    @staticmethod
    def forward(ctx, x, scale, min_threshold, max_threshold):
        x_quant = x * scale
        return x_quant, x_quant

    @staticmethod
    def backward(ctx, grad_x_1, grad_x_2):
        return grad_x_1, grad_x_2, None, None


class OffsetDeQuantizer(OffsetFunction):
    @staticmethod
    def forward(ctx, x, scale, offset, min_threshold, max_threshold):
        x_quant = (x + offset) * scale
        return x_quant, x_quant

    @staticmethod
    def backward(ctx, grad_x_1, grad_x_2):
        return grad_x_1, grad_x_1, grad_x_1, None, None


class Quantizer(NoneOffsetFunction):
    @staticmethod
    def forward(ctx, x, scale, min_threshold, max_threshold):
        x_quant = (x / scale).round().clamp(min_threshold, max_threshold)
        return x_quant, x_quant

    @staticmethod
    def backward(ctx, grad_x_1, grad_x_2):
        return grad_x_1, grad_x_1, None, None


class OffsetQuantizer(OffsetFunction):
    @staticmethod
    def forward(ctx, x, scale, offset, min_threshold, max_threshold):
        x_quant = (x / scale + offset).round().clamp(min_threshold, max_threshold)
        return x_quant, x_quant

    @staticmethod
    def backward(ctx, grad_x_1, grad_x_2):
        return grad_x_1, grad_x_1, grad_x_1, None, None


class PACTQuantizeFunction(NoneOffsetFunction):
    @staticmethod
    def forward(ctx, x, scale, min_threshold, max_threshold):
        ctx.save_for_backward(x, scale)
        ctx.min_threshold, ctx.max_threshold = min_threshold, max_threshold

        x_quant = (x / scale).round().clamp(min_threshold, max_threshold)
        x_dequant = x_quant * scale
        return x_dequant, x_quant

    @staticmethod
    def backward(ctx, grad_hat_output, grad_bar_output):
        x, scale = ctx.saved_tensors
        min_threshold, max_threshold = ctx.min_threshold, ctx.max_threshold

        lower_bound = x < max_threshold
        upper_bound = x > min_threshold
        x_range = ~(lower_bound | upper_bound)

        x_grad = grad_hat_output * x_range
        scale_grad = sum(grad_hat_output * ge(x, max_threshold).float()).view(-1)
        return x_grad, scale_grad, None, None


class STEQuantizeFunction(NoneOffsetFunction):
    @staticmethod
    def forward(ctx, x, scale, min_threshold, max_threshold):
        ctx.save_for_backward(x, scale)
        ctx.min_threshold, ctx.max_threshold = min_threshold, max_threshold

        x_quant = (x / scale).round().clamp(min_threshold, max_threshold)
        x_dequant = x_quant * scale
        return x_dequant, x_quant

    @staticmethod
    def backward(ctx, grad_hat_output, grad_bar_output):
        x, scale = ctx.saved_tensors
        min_threshold, max_threshold = ctx.min_threshold, ctx.max_threshold

        grad_input = grad_hat_output.clone()
        grad_input[x.gt(max_threshold)] = 0
        grad_input[x.lt(min_threshold)] = 0
        return grad_input, grad_input, None, None


class STEOffsetQuantizeFunction(OffsetFunction):
    @staticmethod
    def forward(ctx, x, scale, offset, min_threshold, max_threshold):
        ctx.save_for_backward(x, scale, offset)
        ctx.min_threshold, ctx.max_threshold = min_threshold, max_threshold

        x_quant = (x / scale - offset).round().clamp(min_threshold, max_threshold)
        x_dequant = (x_quant + offset) * scale
        return x_dequant, x_quant

    @staticmethod
    def backward(ctx, grad_hat_output, grad_bar_output):
        x, scale, offset = ctx.saved_tensors
        min_threshold, max_threshold = ctx.min_threshold, ctx.max_threshold

        grad_input = grad_hat_output.clone()
        grad_input[x.gt(max_threshold)] = 0
        grad_input[x.lt(min_threshold)] = 0
        return grad_input, grad_input, grad_input, None, None


class LSQQuantizeFunction(NoneOffsetFunction):
    @staticmethod
    def forward(ctx, x, scale, min_threshold, max_threshold):
        ctx.save_for_backward(x, scale)
        ctx.min_threshold, ctx.max_threshold = min_threshold, max_threshold

        x_quant = (x / scale).round().clamp(min_threshold, max_threshold)
        x_dequant = x_quant * scale
        return x_dequant, x_quant

    @staticmethod
    def backward(ctx, grad_hat_output, grad_bar_output):
        x, scale = ctx.saved_tensors
        min_threshold, max_threshold = ctx.min_threshold, ctx.max_threshold

        grad = 1.0 / sqrt(x.numel() * max_threshold)

        x_q = x / scale
        lower = (x_q <= min_threshold).float()
        higher = (x_q >= max_threshold).float()
        middle = 1.0 - higher - lower

        x_grad = grad_hat_output * middle
        scale_grad = (
            (
                grad_hat_output * lower * min_threshold
                + higher * max_threshold
                + middle * (-x / scale + (x / scale).round()) * grad
            )
            .sum()
            .unsqueeze(dim=0)
        )
        return x_grad, scale_grad, None, None


class LSQPlusQuantizeFunction(OffsetFunction):
    @staticmethod
    def forward(ctx, x, scale, offset, min_threshold, max_threshold):
        ctx.save_for_backward(x, scale, offset)
        ctx.min_threshold, ctx.max_threshold = min_threshold, max_threshold

        x_quant = (x / scale - offset).round().clamp(min_threshold, max_threshold)
        x_dequant = (x_quant + offset) * scale
        return x_dequant, x_quant

    @staticmethod
    def backward(ctx, grad_hat_output, grad_bar_output):
        x, scale, offset = ctx.saved_tensors
        min_threshold, max_threshold = ctx.min_threshold, ctx.max_threshold

        grad = 1.0 / sqrt(x.numel() * max_threshold)

        x_q = x / scale - offset
        lower = (x_q <= min_threshold).float()
        higher = (x_q >= max_threshold).float()
        middle = 1.0 - higher - lower

        x_grad = grad_hat_output * middle
        scale_grad = (
            (
                grad_hat_output * lower * min_threshold
                + higher * max_threshold
                + middle * (-x / scale + (x / scale).round()) * grad
            )
            .sum()
            .unsqueeze(dim=0)
        )
        offset_grad = ((lower + higher) * grad_hat_output * grad).sum().unsqueeze(dim=0)
        return x_grad, scale_grad, offset_grad, None, None
