from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor, flatten, float, kthvalue
from torch import max as torch_max
from torch import min as torch_min
from torch import tensor
from torch.nn import Module


def kth_value_minimum(x: Tensor, percentile: float):
    return kthvalue(flatten(x), max(1, min(x.numel(), int(x.numel() * percentile))))[0]


def kth_value_maximum(x: Tensor, percentile: float):
    return kthvalue(flatten(x), min(x.numel(), max(1, int(x.numel() * (1 - percentile)))))[0]


class RangeObserver(Module, ABC):
    """
    Base vision quantizer that use define quantize function for specific range based on number of bits
    """

    def __init__(
        self,
        bits: int,
    ):
        """
        :param bits: target number of bits to quantize data inside its range
        """
        super(RangeObserver, self).__init__()
        self.bits = bits

    @abstractmethod
    def observe(self, data):
        """
        get the configuration for the range observer
        """
        raise NotImplementedError(f"{self.__class__.__name__} suppose to be abstract class")

    @abstractmethod
    def get_ranger_configuration(self):
        """
        get the configuration for the range observer
        """
        raise NotImplementedError(f"{self.__class__.__name__} suppose to be abstract class")

    @abstractmethod
    def get_rangers(self) -> Tuple[int, int]:
        """
        get the minimum and maximum values for the range
        """
        raise NotImplementedError(f"{self.__class__.__name__} suppose to be abstract class")


class UniformRangeObserver(RangeObserver):
    """
    Uniform quantizer to quantize the values with the same gap between each data point
    """

    def __init__(
        self,
        bits: int,
        is_positive: bool = False,
        is_symmetric: bool = False,
    ):
        """
        :param is_positive: True if the quantizer suppose to have only positive values, otherwise False
        :param is_symmetric: True if the quantized values need to symmetric around zero, otherwise False
        """
        super(UniformRangeObserver, self).__init__(bits)

        self.bits = bits
        self.is_positive = is_positive
        self.is_symmetric = is_symmetric

        self.auxiliaries_dict = {}
        self.quantizer_configuration_dict = {
            "bits": bits,
            "is_positive": is_positive,
            "is_symmetric": is_symmetric,
        }

        if is_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.min_threshold = 0
            self.max_threshold = 2 ** self.bits - 1
        else:
            if is_symmetric:
                # symmetric signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.min_threshold = -(2 ** (self.bits - 1)) + 1
                self.max_threshold = 2 ** (self.bits - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.min_threshold = -(2 ** (self.bits - 1))
                self.max_threshold = 2 ** (self.bits - 1) - 1

        self.range_magnitude = abs((self.max_threshold - self.min_threshold))

    def observe(self, data):
        return False

    def get_ranger_configuration(self):
        return self.quantizer_configuration_dict

    def get_rangers(self):
        return self.min_threshold, self.max_threshold


class MinMaxUniformRangeObserver(UniformRangeObserver):
    """
    Compute the minimum and maximum for batches to quantizer for quantizing the values based on the ranges
    """

    def __init__(
        self,
        bits: int,
        is_positive: bool = False,
        is_symmetric: bool = False,
        percentile: float = None,
    ):
        """
        :param percentile: percentile to cut off the maximum, and minimum values based on k-th values
        """
        super(MinMaxUniformRangeObserver, self).__init__(bits, is_positive, is_symmetric)
        if percentile is None:
            self.minimum_function = torch_min
            self.maximum_function = torch_max
        else:
            self.minimum_function = kth_value_minimum
            self.maximum_function = kth_value_maximum

    def observe(self, data):
        min_value, max_value = (
            self.minimum_function(data).round().detach(),
            self.maximum_function(data).round().detach(),
        )
        if min_value == max_value:
            self.min_threshold = self.min_threshold + min_value
            self.max_threshold = self.max_threshold + max_value
        else:
            self.min_threshold = min(self.min_threshold, min_value)
            self.max_threshold = max(self.max_threshold, max_value)
        return True


class MomentumMinMaxUniformRangeObserver(MinMaxUniformRangeObserver):
    """
    Momentum to update the minimum, and maximum ranges by shifting the ranges using small fraction
    """

    def __init__(
        self,
        bits: int,
        is_positive: bool = False,
        is_symmetric: bool = False,
        percentile: float = None,
        momentum: float = 0.9,
    ):
        """
        :param momentum: positive fraction to shift the ranges for quantizer
        """
        super(MomentumMinMaxUniformRangeObserver, self).__init__(bits, is_positive, is_symmetric, percentile)
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        self.momentum = tensor(momentum, requires_grad=False)

    def observe(self, data):
        min_value, max_value = (
            self.minimum_function(data).round().detach(),
            self.maximum_function(data).round().detach(),
        )
        if min_value == max_value:
            self.min_threshold = self.min_threshold + min_value
            self.max_threshold = self.max_threshold + max_value
        else:
            self.min_threshold = (self.min_threshold * self.momentum + min_value * (1 - self.momentum)).round()
            self.max_threshold = (self.max_threshold * self.momentum + max_value * (1 - self.momentum)).round()
        return True
