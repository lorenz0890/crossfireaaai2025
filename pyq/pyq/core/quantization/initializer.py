import abc
import warnings
from abc import ABC, abstractmethod
from math import sqrt
from typing import Any

from torch import Tensor, clamp, finfo, float32, max, min, tensor
from torch.distributions import Normal
from torch.nn import Module, Parameter

from pyq.core.quantization.observer import RangeObserver

DEFAULT_INITIALIZER_NAME = "initializer"


class Initializer(Module, ABC):
    def __init__(self):
        super(Initializer, self).__init__()

    def build(self, data: Any):
        """
        Return the parameters as directory with the name to be registered as key and the initialization value as value
        """
        raise NotImplementedError(f"{self.__class__.__name__} suppose to be abstract class")

    @abstractmethod
    def get_learned_parameters(self):
        """
        Return the parameters as directory with the name to be registered as key and the initialization value as value
        """
        raise NotImplementedError(f"{self.__class__.__name__} suppose to be abstract class")

    @abstractmethod
    @abc.abstractmethod
    def get_parameters_initial_value(self) -> dict:
        """
        Return the parameters as directory with the name to be registered as key and the initialization value as value
        """
        raise NotImplementedError(f"{self.__class__.__name__} suppose to be abstract class")


class DictionaryToEmptyInitializer(Initializer):
    def __init__(self, dictionary: dict):
        super(DictionaryToEmptyInitializer, self).__init__()
        self.dictionary = dictionary
        for key in dictionary:
            setattr(self, key, dictionary[key].detach() if isinstance(dictionary[key], Tensor) else dictionary[key])
        self.is_initialized = True

    def build(self, *args: Any, **kwargs: Any):
        return

    def set_range_observer(self, *args: Any, **kwargs: Any):
        return

    def get_learned_parameters(self) -> dict:
        return self.dictionary

    def get_parameters_initial_value(self) -> dict:
        return self.dictionary


class QuantizationInitializer(Initializer):
    """
    Base class to initialize the quantizer learnable parameters, only initialize one time.
    Initializer must be child from module, and the parameters should assign in __init__ to obtain gradient for them.
    """

    def __init__(self):
        super(QuantizationInitializer, self).__init__()
        # buffer is not updated for optim.step
        self.is_initialized = False
        parameters_initialize_values = self.get_parameters_initial_value()
        for param_name in parameters_initialize_values:
            setattr(self, param_name, Parameter(Tensor([parameters_initialize_values[param_name]])))

    def __set_arguments__(self, **kwargs: Any):
        """
        Accept any attributes with names, and values then create the attributes inside the initializer scope
        """
        _ = [setattr(self, argument_name, kwargs[argument_name]) for argument_name in kwargs]

    def set_range_observer(self, observer: RangeObserver):
        """
        Set the range observer that contain the ranges and also contain learnable parameters for this range
        """
        # TODO (Samir): move this to the `__init__` to be able to learn the learnable parameters for this range
        self.observer = observer
        self.min_threshold, self.max_threshold = observer.get_rangers()

    def build(self, x):
        """
        initialize and register parameters for the first attempt when calling the quantizer
        """
        # skip the initialization, and the registration of parameters if it's already initialized
        if not self.is_initialized:
            self.__set_arguments__(
                data=tensor(x, requires_grad=False),
                negative_threshold=self.min_threshold,
                positive_threshold=self.max_threshold,
                **self.observer.get_ranger_configuration(),
            )
            self.auxiliaries_dict = self.get_parameters_initial_value()

            for aux_key in self.auxiliaries_dict.keys():
                if aux_key in self.state_dict().keys():
                    aux_value = self.auxiliaries_dict[aux_key]
                    aux_value = aux_value if isinstance(aux_value, Tensor) else tensor(aux_value, requires_grad=True)
                    getattr(self, aux_key).data.copy_(aux_value)
                else:
                    self.register_parameter(aux_key, Parameter(self.auxiliaries_dict[aux_key]))

            self.is_initialized = True
        else:
            warnings.warn("parameters were set already with the desired values, it will override now with new values")

    def __is_input_data_fitted__(self, defaults):
        defaults_tensor_dict = {name: tensor(defaults[name], requires_grad=True) for name in defaults}
        if not hasattr(self, "data") or self.data is None:
            return False, defaults_tensor_dict
        absolute_mean = self.data.abs().mean()
        if absolute_mean == 0:
            return False, defaults_tensor_dict
        return True, None

    def get_learned_parameters(self):
        initialize_parameters_names = list(self.defaults.keys()) + ["min_threshold", "max_threshold"]
        return {name: getattr(self, name) for name in initialize_parameters_names}


class RandomInitializer(QuantizationInitializer):
    def __init__(self):
        self.defaults = {"scale": Normal(1, 0.1).sample([1])}
        super(RandomInitializer, self).__init__()

    def get_parameters_initial_value(self):
        return self.defaults


class RandomOffsetInitializer(QuantizationInitializer):
    def __init__(self):
        self.defaults = {"scale": Normal(1, 0.1).sample([1]), "offset": Normal(0, 0.1).sample([1])}
        super(RandomOffsetInitializer, self).__init__()

    def get_parameters_initial_value(self):
        return self.defaults


class FixedInitializer(QuantizationInitializer):
    def __init__(self):
        self.defaults = {"scale": 1.0}
        super(FixedInitializer, self).__init__()

    def get_parameters_initial_value(self):
        return self.defaults


class FixedOffsetInitializer(QuantizationInitializer):
    def __init__(self):
        self.defaults = {"scale": 1.0, "offset": 0.0}
        super(FixedOffsetInitializer, self).__init__()

    def get_parameters_initial_value(self):
        return self.defaults


class MaxMinInitializer(QuantizationInitializer):
    """
    source Parameterized Clipping Activation: https://arxiv.org/abs/1805.06085
    """

    def __init__(self):
        self.defaults = {"scale": 1.0}
        super(MaxMinInitializer, self).__init__()

    def get_parameters_initial_value(self):
        is_input_data_fitted, default_parameters = self.__is_input_data_fitted__(self.defaults)
        if not is_input_data_fitted:
            return default_parameters

        min_value = self.data.min()
        max_value = self.data.max()
        scale = (self.max_threshold - self.min_threshold) / (max_value - min_value)
        return {"scale": scale}


class MaxMinOffsetInitializer(QuantizationInitializer):
    """
    source Parameterized Clipping Activation: https://arxiv.org/abs/1805.06085
    """

    def __init__(self):
        self.defaults = {"scale": 1.0, "offset": 0.0}
        super(MaxMinOffsetInitializer, self).__init__()

    def get_parameters_initial_value(self):
        is_input_data_fitted, default_parameters = self.__is_input_data_fitted__(self.defaults)
        if not is_input_data_fitted:
            return default_parameters

        min_value = self.data.min()
        max_value = self.data.max()
        scale = (self.max_threshold - self.min_threshold) / (max_value - min_value)
        scale = max(scale, tensor([finfo(float32).eps], device=self.data.device))
        offset = self.min_threshold - (min_value / scale).round()
        offset = max(tensor([self.min_threshold], device=self.data.device), offset)
        offset = min(tensor([self.max_threshold], device=self.data.device), offset)
        offset = clamp(offset, self.min_threshold, self.max_threshold)
        return {"scale": scale, "offset": offset}


class MinMaxInitializer(QuantizationInitializer):
    """
    source Parameterized Clipping Activation: https://arxiv.org/abs/1805.06085
    """

    def __init__(self):
        self.defaults = {"scale": 1.0}
        super(MinMaxInitializer, self).__init__()

    def get_parameters_initial_value(self):
        is_input_data_fitted, default_parameters = self.__is_input_data_fitted__(self.defaults)
        if not is_input_data_fitted:
            return default_parameters

        min_value = self.data.min()
        max_value = self.data.max()
        scale = (max_value - min_value) / (self.max_threshold - self.min_threshold)
        return {"scale": scale}


class MinMaxOffsetInitializer(QuantizationInitializer):
    """
    source Parameterized Clipping Activation: https://arxiv.org/abs/1805.06085
    """

    def __init__(self):
        self.defaults = {"scale": 1.0, "offset": 0.0}
        super(MinMaxOffsetInitializer, self).__init__()

    def get_parameters_initial_value(self):
        is_input_data_fitted, default_parameters = self.__is_input_data_fitted__(self.defaults)
        if not is_input_data_fitted:
            return default_parameters

        min_value = self.data.min()
        max_value = self.data.max()
        scale = (max_value - min_value) / (self.max_threshold - self.min_threshold)
        scale = max(scale, tensor([finfo(float32).eps], device=self.data.device))
        offset = self.min_threshold - (min_value / scale).round()
        offset = max(tensor([self.min_threshold], device=self.data.device), offset)
        offset = min(tensor([self.max_threshold], device=self.data.device), offset)
        offset = clamp(offset, self.min_threshold, self.max_threshold)
        return {"scale": scale, "offset": offset}


class PACTInitializer(QuantizationInitializer):
    """
    source Parameterized Clipping Activation: https://arxiv.org/abs/1805.06085
    """

    def __init__(self):
        self.defaults = {"scale": 1.0}
        super(PACTInitializer, self).__init__()

    def get_parameters_initial_value(self):
        is_input_data_fitted, default_parameters = self.__is_input_data_fitted__(self.defaults)
        if not is_input_data_fitted:
            return default_parameters

        scale = 2 * self.data.abs().mean() / sqrt(self.max_threshold)
        return {"scale": scale}


class PACTOffsetInitializer(QuantizationInitializer):
    """
    source Parameterized Clipping Activation: https://arxiv.org/abs/1805.06085
    """

    def __init__(self):
        self.defaults = {"scale": 1.0, "offset": 0.0}
        super(PACTOffsetInitializer, self).__init__()

    def get_parameters_initial_value(self):
        is_input_data_fitted, default_parameters = self.__is_input_data_fitted__(self.defaults)
        if not is_input_data_fitted:
            return default_parameters

        # min_value = self.data.min()
        scale = 2 * self.data.abs().mean() / sqrt(self.max_threshold)
        offset = 0.0
        return {"scale": scale, "offset": offset}


class LsqInitializer(QuantizationInitializer):
    """
    source Learned Step Size Quantization (LSQ): https://arxiv.org/abs/1902.08153
    """

    def __init__(self):
        self.defaults = {"scale": 1.0}
        super(LsqInitializer, self).__init__()

    def get_parameters_initial_value(self):
        is_input_data_fitted, default_parameters = self.__is_input_data_fitted__(self.defaults)
        if not is_input_data_fitted:
            return default_parameters

        absolute_mean = self.data.abs().mean()
        scale = 2 * absolute_mean / sqrt(2 ** (self.bits - 1) - 1)
        return {"scale": scale}


class LsqPlusInitializer(QuantizationInitializer):
    """
    source LSQ+: Improving low-bit quantization through learnable offsets and better initialization:
    https://arxiv.org/abs/2004.09576
    """

    def __init__(self):
        self.defaults = {"scale": 1.0, "offset": 0.0}
        super(LsqPlusInitializer, self).__init__()

    def get_parameters_initial_value(self):
        is_input_data_fitted, default_parameters = self.__is_input_data_fitted__(self.defaults)
        if not is_input_data_fitted:
            return default_parameters

        mean = self.data.mean()
        std = self.data.std()
        lower_bound, upper_bound = (mean - 3 * std).abs(), (mean + 3 * std).abs()
        scale = max(lower_bound, upper_bound) / (2 ** (self.bits - 1))
        offset = self.data.max() - self.bits * scale
        return {"scale": scale, "offset": offset}
