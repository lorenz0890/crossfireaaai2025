import abc
import operator
import warnings
from typing import Any, Callable, Optional, Tuple, Union

from torch import Tensor, count_nonzero, numel, tensor
from torch.nn import Module, ModuleDict, Parameter, ParameterDict

from pyq.core.quantization.functional import (DEFAULT_CTX_NAME, DeQuantizer, IdentityFunction, ModuleFunction,
                                              NoneOffsetFunction, OffsetDeQuantizer, OffsetQuantizer, PyQFunction,
                                              Quantizer)
from pyq.core.quantization.initializer import (DEFAULT_INITIALIZER_NAME, DictionaryToEmptyInitializer, Initializer,
                                               QuantizationInitializer)
from pyq.core.quantization.observer import RangeObserver
from pyq.core.quantization.supported_operations import (DEFAULT_ATTRIBUTES_NAME, DEFAULT_OPERATION_NAME,
                                                        layer_to_operation)
from pyq.core.wrapper import DEFAULT_WRAPPED_OBJECT_NAME, TorchModuleWrapper
from pyq.datasets.initializer import X_NAME
from pyq.experiment.function_builder import __intersection_between_function_argument_names_and_instance_variables__
from pyq.models.editor import DOT, setattr_nested

UNDERSCORE = "_"
IS = "is"

DEFAULT_WEIGHT_NAME = "weight"
DEFAULT_BIAS_NAME = "bias"

DEFAULT_ARGUMENTS_NAME = "arguments"
DEFAULT_PARAMETERS_NAME = "parameters"
DEFAULT_ACTIVATION_NAME = "activation"

DEFAULT_AUXILIARY_NAME = "auxiliary"
DEFAULT_QUANTIZED_NAME = "quantized"
DEFAULT_MODEL_BIT_WIDTH_NAME = "model_bit_width"
DEFAULT_MODEL_SIZE_NAME = "model_size"
DEFAULT_MODEL_FULL_SIZE_NAME = "model_full_size"

DEFAULT_QUANTIZED_WEIGHT_NAME = UNDERSCORE.join([DEFAULT_QUANTIZED_NAME, DEFAULT_WEIGHT_NAME])
DEFAULT_QUANTIZED_BIAS_NAME = UNDERSCORE.join([DEFAULT_QUANTIZED_NAME, DEFAULT_BIAS_NAME])

IS_QUANTIZED = UNDERSCORE.join([IS, DEFAULT_QUANTIZED_NAME])
AUXILIARY_IS_QUANTIZED = UNDERSCORE.join([DEFAULT_AUXILIARY_NAME, IS_QUANTIZED])


def __preform_tensor_mapping__(
    value: Tensor, instance_with_arguments: Any, mapping_function: Union[PyQFunction, ModuleFunction]
) -> Tuple[Tensor, Tensor]:
    """
    Extract the needed arguments names from the quantization function and then extract these names values from
    the current object or from the initializer.
    :param value: value that needed to be quantized
    :param instance_with_arguments: responsible for setting the initial values for learnable quantization parameter
    :param mapping_function: main function that used to quantize the values

    :return: dequantized value, and quantized value that computed using learnable parameters
    """
    # get the values to pass them to the quantizer function
    # remove the `x`, and `ctx` argument names from the quantizer arguments dictionary
    mapper_arguments_dict = __intersection_between_function_argument_names_and_instance_variables__(
        function=mapping_function, instance=instance_with_arguments, names_to_skip=[X_NAME, DEFAULT_CTX_NAME]
    )
    # concatenate the value that need to be quantized the learnable parameter
    mapper_arguments_list = [value] + [*mapper_arguments_dict.values()]
    # apply the quantization using the `quantize_function`
    mapped_value, demapped_value = mapping_function.apply(*mapper_arguments_list)
    return mapped_value, demapped_value


# TODO(Samir) 1: decompose this class to be InputQuantizer, and LayerQuantizer
#              `InputQuantizer` for ActivationQuantizer, and LayerQuantizer for the rest of the layers.

# TODO(Samir) 2: Absorber the scales during the unwrapping to be inbetween the min, max ranges


class LayerQuantizerWrapper(TorchModuleWrapper):
    """
    Base layer wrapper that inherent from torch wrapper to quantize layer parameter, each task should have its own
    quantizer wrapper that inherent from this class and override every method in this class.
    """

    def __init__(
        self,
        torch_object: Module,
        range_observer: RangeObserver,
        initializer: QuantizationInitializer,
        quantizer_function: PyQFunction,
        is_quantize_input: bool = True,
        is_quantize_parameter: bool = True,
        is_quantize_output: bool = True,
    ):
        """
        :param torch_object: torch module instance object to wrap it
        :param range_observer: quantizer to quantize the weight and bias if it exists in the wrapped layer
        :param initializer: initialization values and register parameters for the quantization
        :param quantizer_function: quantization function to discretize the continuous values of the parameters
        """
        super(LayerQuantizerWrapper, self).__init__(torch_object)

        self.__initializer__ = initializer
        self.__quantizer_function__ = quantizer_function
        self.__range_observer__ = range_observer

        self.device = [*initializer.parameters()][0].device

        # set weight quantizer functions
        self.set_input_to_quantize(is_quantize_input)
        self.set_parameter_to_quantize(is_quantize_parameter)
        self.set_output_to_quantize(is_quantize_output)

        # set flags for the quantization parts
        self.is_quantize_input = is_quantize_input
        self.is_quantize_parameter = is_quantize_parameter
        self.is_quantize_output = is_quantize_output

        # set `is_quantized` flag with false value
        self.register_non_persistent_buffer_per_layer(IS_QUANTIZED, False)

    def set_input_to_quantize(self, is_quantize_input: bool):
        self.is_quantize_input = is_quantize_input
        if is_quantize_input:
            # set weight quantizer functions
            self.__set_input_initializer__(self.__initializer__, self.__range_observer__)
            self.__set_input_quantizer_function__(self.__quantizer_function__)
            self.__set_input_range_observer__(self.__range_observer__)

    def set_parameter_to_quantize(self, is_quantize_parameter: bool):
        self.is_quantize_parameter = is_quantize_parameter
        if is_quantize_parameter:
            # set weight quantizer functions
            self.__set_parameter_initializer__(self.__initializer__, self.__range_observer__)
            self.__set_parameter_quantizer_function__(self.__quantizer_function__)
            self.__set_parameter_range_observer__(self.__range_observer__)

    def set_output_to_quantize(self, is_quantize_output: bool):
        self.is_quantize_output = is_quantize_output
        if is_quantize_output:
            # set weight quantizer functions
            self.__set_output_initializer__(self.__initializer__, self.__range_observer__)
            self.__set_output_quantizer_function__(self.__quantizer_function__)
            self.__set_output_range_observer__(self.__range_observer__)

    def __set_input_quantizer_function__(self, quantizer_function: Union[PyQFunction, ModuleFunction]):
        """
        Setter for the quantizer function in the quantizer wrapper
        """
        self.input_quantizer_function = quantizer_function

    def __set_parameter_quantizer_function__(self, quantizer_function: Union[PyQFunction, ModuleFunction]):
        """
        Setter for the quantizer function in the quantizer wrapper
        """
        self.parameter_quantizer_function = quantizer_function

    def __set_output_quantizer_function__(self, dequantizer_function: Union[PyQFunction, ModuleFunction]):
        """
        Setter for the quantizer function in the quantizer wrapper
        """
        self.output_quantizer_function = dequantizer_function

    def __set_input_range_observer__(self, range_observer: RangeObserver):
        """
        Setter for the range observer in the quantizer wrapper
        """
        # initiate new instance for each layer for `range_observer`
        self.input_range_observer = range_observer.__class__(**range_observer.get_ranger_configuration()).to(
            self.device
        )

    def __set_parameter_range_observer__(self, range_observer: RangeObserver):
        """
        Setter for the range observer in the quantizer wrapper
        """
        # initiate new instance for each layer for `range_observer`
        self.parameter_range_observer = range_observer.__class__(**range_observer.get_ranger_configuration()).to(
            self.device
        )

    def __set_output_range_observer__(self, range_observer: RangeObserver):
        """
        Setter for the range observer in the quantizer wrapper
        """
        # initiate new instance for each layer for `range_observer`
        self.output_range_observer = range_observer.__class__(**range_observer.get_ranger_configuration()).to(
            self.device
        )

    def __set_input_initializer__(
        self, initializer: Union[QuantizationInitializer, DictionaryToEmptyInitializer], range_observer: RangeObserver
    ):
        """
        Setter for the initializer in the quantizer wrapper
        """
        # initiate new instance for each layer for `initializer`
        if isinstance(initializer, QuantizationInitializer):
            initializer = initializer.__class__().to(self.device)
            initializer.set_range_observer(range_observer)
        self.input_initializer = initializer

    def __set_parameter_initializer__(
        self, initializer: Union[QuantizationInitializer, DictionaryToEmptyInitializer], range_observer: RangeObserver
    ):
        """
        Setter for the initializer in the quantizer wrapper
        """
        # initiate new instance for each layer for `initializer`
        if isinstance(initializer, QuantizationInitializer):
            initializer = initializer.__class__().to(self.device)
            initializer.set_range_observer(range_observer)
        self.parameter_initializer = initializer

    def __set_output_initializer__(
        self, initializer: Union[QuantizationInitializer, DictionaryToEmptyInitializer], range_observer: RangeObserver
    ):
        """
        Setter for the initializer in the quantizer wrapper
        """
        # initiate new instance for each layer for `initializer`
        if isinstance(initializer, QuantizationInitializer):
            initializer = initializer.__class__().to(self.device)
            initializer.set_range_observer(range_observer)
        self.output_initializer = initializer

    def __pre_quantization__(
        self, initializer: Initializer, function: PyQFunction, range_observer: RangeObserver, data: Tensor
    ) -> Tuple[Initializer, RangeObserver]:
        """
        Build the initialization  for a quantizer and register the parameters for it.
        """
        # initialization the weight quantizer if it's not initialized
        initializer.build(data)
        quantized_data = __preform_tensor_mapping__(data, initializer, function)
        range_observer.observe(quantized_data)
        initializer.set_range_observer(range_observer)
        return initializer, range_observer

    def register_non_persistent_buffer_per_layer(self, name: str, value: Union[bool, int, float, list, tuple]) -> None:
        """
        Register a buffer parameter in the layer to recall its value later
        :param name: name of the parameter as string
        :param value: value to for the parameter to store
        """
        # convert the value to tensor, and then store it as buffer in the layer
        self.register_buffer(UNDERSCORE.join([DEFAULT_AUXILIARY_NAME, name]), tensor(value), persistent=False)

    @abc.abstractmethod
    def __update_layer_parameters__(self):
        """
        Update the wrapped layer parameter or inject new parameters
        """
        """
                del self.__initializer__
                del self.__quantizer_function__
                del self.__range_observer__
        """

        raise NotImplementedError(f"{self.__class__.__name__} suppose to be abstract class")

    @abc.abstractmethod
    def __quantize__(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Quantization method that use the quantizer/s to obtain specific `Tensor` which is quantized or dequantized
        """
        raise NotImplementedError(f"{self.__class__.__name__} suppose to be abstract class")

    @abc.abstractmethod
    def __post_quantization__(self, *args: Any, **kwargs: Any):
        """
        Optional postprocessing step that suppose to execute after finishing the whole quantization
        """
        raise NotImplementedError(f"{self.__class__.__name__} suppose to be abstract class")


class ActivationQuantizerWrapper(LayerQuantizerWrapper):
    """
    Basic wrapper for quantization of the activation functions by quantize the output from previous layer;
    Q(x_{i+1}) = W . Q_{x}(x_{i}) + b
    """

    def __init__(
        self,
        torch_object: Module,
        range_observer: RangeObserver,
        initializer: QuantizationInitializer,
        quantizer_function: PyQFunction,
        is_quantize_input: bool = False,
        is_quantize_parameter: bool = False,
        is_quantize_output: bool = True,
    ):
        assert (
            not is_quantize_parameter
        ), "{} suppose to quantize activation function is parameters should be inside the layer {}".format(
            self.__class__.__name__, torch_object.__class__.__name__
        )
        super(ActivationQuantizerWrapper, self).__init__(
            torch_object,
            range_observer,
            initializer,
            quantizer_function,
            is_quantize_input,
            is_quantize_parameter,
            is_quantize_output,
        )

    def __update_layer_parameters__(self):
        """
        Update the quantizer to be just identy the input data.
        """

        if self.is_quantize_input:
            updated_input_quantizer = (
                DeQuantizer() if isinstance(self.input_quantizer_function, NoneOffsetFunction) else OffsetDeQuantizer()
            )
            self.__set_input_quantizer_function__(updated_input_quantizer)
            non_learnable_initializer = DictionaryToEmptyInitializer(self.input_initializer.get_learned_parameters())
            self.__set_parameter_initializer__(non_learnable_initializer, self.input_range_observer)
            self.__quantizer_function__ = updated_input_quantizer

        if self.is_quantize_output:
            updated_output_quantizer = (
                Quantizer() if isinstance(self.output_quantizer_function, NoneOffsetFunction) else OffsetQuantizer()
            )
            self.__set_output_quantizer_function__(updated_output_quantizer)
            non_learnable_initializer = DictionaryToEmptyInitializer(self.output_initializer.get_learned_parameters())
            self.__set_output_initializer__(non_learnable_initializer, self.output_range_observer)

        self.__quantizer_function__ = IdentityFunction()
        [delattr(self, name) for name in self.state_dict().keys() if name.startswith(DEFAULT_AUXILIARY_NAME)]

    def __post_quantization__(self):
        """
        Preform the `update_layer` and `del_auxiliary` functions
        """
        # check that layer has wrapped object, if not skip unwrapping this layer
        if not hasattr(self, DEFAULT_WRAPPED_OBJECT_NAME) or not isinstance(self, LayerQuantizerWrapper):
            return

        self.__update_layer_parameters__()

    def __call__(self, x):
        """
        Override the feedforward call to quantize the parameters and then call the layer feedforward
        to have complete differentiable quantization pipeline
        """
        if self.is_quantize_input:
            self.input_initializer, self.input_range_observer = self.__pre_quantization__(
                self.input_initializer, self.input_quantizer_function, self.input_range_observer, x
            )
            x, x_q = __preform_tensor_mapping__(x, self.input_initializer, self.input_quantizer_function)

        x = super(ActivationQuantizerWrapper, self).__call__(x)

        if self.is_quantize_output:
            self.output_initializer, self.output_range_observer = self.__pre_quantization__(
                self.output_initializer, self.output_quantizer_function, self.output_range_observer, x
            )
            x, x_q = __preform_tensor_mapping__(x, self.output_initializer, self.output_quantizer_function)
        return x

    @staticmethod
    def unwrap(obj):
        """
        Unwrap the activation function without storing any quantized value, but store quantization parameter e.g. scale

        :param obj: object to unwrap
        :return: unwrapped object
        """

        ActivationQuantizerWrapper.__post_quantization__(obj)
        return obj


class TransformationLayerQuantizerWrapper(LayerQuantizerWrapper):
    """
    Basic wrapper for quantization of linear transformation tasks `x_{i+1} = W.x_{i} + b` by quantize weights
     (W), and bias (b) if they're exist; Q(x_{i+1}) = Q_{w}(W) . x_{i} + Q_{w}(b)
    """

    def __init__(
        self,
        torch_object: Module,
        range_observer: RangeObserver,
        initializer: QuantizationInitializer,
        quantizer_function: PyQFunction,
        is_quantize_input: bool = False,
        is_quantize_parameter: bool = True,
        is_quantize_output: bool = True,
    ):
        super(TransformationLayerQuantizerWrapper, self).__init__(
            torch_object,
            range_observer,
            initializer,
            quantizer_function,
            is_quantize_input,
            is_quantize_parameter,
            is_quantize_output,
        )

        assert type(torch_object) in layer_to_operation, "Unsupported layer operation {}.".format(type(torch_object))

    def __quantize_weight_and_bias__(
        self, weight: Tensor, bias: Optional[Tensor] = None
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Quantize the weight, and bias "if found" of the layer based on the quantizers that provided to the constractor
        """
        # weight quantization
        dequantized_weight, quantized_weight = __preform_tensor_mapping__(
            weight, self.parameter_initializer, self.parameter_quantizer_function
        )

        # biases quantization
        dequantized_bias, quantized_bias = None, None
        if bias is not None:
            dequantized_bias, quantized_bias = __preform_tensor_mapping__(
                bias, self.parameter_initializer, self.parameter_quantizer_function
            )
        return (dequantized_weight, quantized_weight), (dequantized_bias, quantized_bias)

    def __update_layer_parameters__(self):
        """
        Update the weight and bias "if found" for the layer with their quantized values, and update the quantization
        function with dequantize function.
        """
        if self.is_quantize_input:
            updated_input_quantizer = (
                Quantizer() if isinstance(self.input_quantizer_function, NoneOffsetFunction) else OffsetQuantizer()
            )
            self.__set_input_quantizer_function__(updated_input_quantizer)
            non_learnable_initializer = DictionaryToEmptyInitializer(self.input_initializer.get_learned_parameters())
            self.__set_parameter_initializer__(non_learnable_initializer, self.input_range_observer)

        if self.is_quantize_parameter:
            self._wrapped_object.weight = Parameter(self.quantized_weight)
            self._wrapped_object.bias = Parameter(self.quantized_bias)

            non_learnable_initializer = DictionaryToEmptyInitializer(
                self.parameter_initializer.get_learned_parameters()
            )
            self.__set_parameter_quantizer_function__(IdentityFunction())
            self.__set_parameter_initializer__(non_learnable_initializer, self.parameter_range_observer)

        if self.is_quantize_output:
            updated_output_quantizer = (
                Quantizer() if isinstance(self.output_quantizer_function, NoneOffsetFunction) else OffsetQuantizer()
            )
            self.__set_output_quantizer_function__(updated_output_quantizer)
            non_learnable_initializer = DictionaryToEmptyInitializer(self.output_initializer.get_learned_parameters())
            self.__set_output_initializer__(non_learnable_initializer, self.output_range_observer)

        [delattr(self, name) for name in self.state_dict().keys() if name.startswith(DEFAULT_AUXILIARY_NAME)]

    def __post_quantization__(self):
        """
        Check the wrapping, and then apply `update_layer`, and `del_auxiliary`
        """
        # check that layer has wrapped object, if not skip unwrapping this layer
        if not isinstance(self, LayerQuantizerWrapper):
            return

        if not hasattr(self, AUXILIARY_IS_QUANTIZED):
            warnings.warn("{} was applied as wrapper, but no quantization happen.".format(self))
            return

        if hasattr(self, AUXILIARY_IS_QUANTIZED) and not getattr(self, AUXILIARY_IS_QUANTIZED):
            warnings.warn(
                "{} doesn't preform the feedforward, or have no parameter to quantize,try to increase "
                "num_layers.".format(self)
            )
            self = getattr(self, DEFAULT_WRAPPED_OBJECT_NAME)
            return

        self.__update_layer_parameters__()

    def __get_operation_function_and_operation_attributes__(self) -> Tuple[Callable, dict]:
        operation_dict = layer_to_operation[type(self._wrapped_object)]
        operation = operation_dict[DEFAULT_OPERATION_NAME]
        operation_names_attributes = operation_dict[DEFAULT_ATTRIBUTES_NAME]
        operation_attributes_dict = {
            attribute: getattr(self._wrapped_object, attribute) for attribute in operation_names_attributes
        }
        return operation, operation_attributes_dict

    @property
    def __has_bias__(self):
        return (
            hasattr(self._wrapped_object, "bias")
            and self._wrapped_object.bias is not None
            and self._wrapped_object.bias.numel() > 0
        )

    def __call__(self, x):
        """
        Override the feedforward call to quantize the parameters and then call the layer feedforward
        to have complete differentiable quantization pipeline
        """

        if self._forward_pre_hooks:
            for hook_id, hook in (*self._forward_pre_hooks.items(),):
                #print('HOOK ID', hook_id, flush=True)
                x = hook(self, x)

            #exit(-1)
        self.register_non_persistent_buffer_per_layer(IS_QUANTIZED, True)
        self.register_non_persistent_buffer_per_layer(DEFAULT_MODEL_BIT_WIDTH_NAME, self.parameter_range_observer.bits)

        # preform input quantization
        if self.is_quantize_input:
            self.input_initializer, self.input_range_observer = self.__pre_quantization__(
                self.input_initializer, self.input_quantizer_function, self.input_range_observer, x
            )
            x, x_q = __preform_tensor_mapping__(x, self.input_initializer, self.input_quantizer_function)

        weight = self._wrapped_object.weight
        bias = self._wrapped_object.bias if self.__has_bias__ else None

        # preform parameter quantization
        if self.is_quantize_parameter:
            self.parameter_initializer, self.parameter_range_observer = self.__pre_quantization__(
                self.parameter_initializer, self.parameter_quantizer_function, self.parameter_range_observer, weight
            )
            (weight, quantized_weight), (bias, quantized_bias) = self.__quantize_weight_and_bias__(weight, bias)

            # store quantize parameters to construct quantized model
            self.quantized_weight = quantized_weight
            self.quantized_bias = quantized_bias if self.__has_bias__ else None

        # count non-zero values and multiply them by the precision of the layer
        physical_size = count_nonzero(weight) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(DEFAULT_MODEL_SIZE_NAME, physical_size)

        # count non-zero values and multiply them by the precision of the layer
        full_physical_size = numel(weight) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(DEFAULT_MODEL_FULL_SIZE_NAME, full_physical_size)

        operation, operation_attributes_dict = self.__get_operation_function_and_operation_attributes__()

        # extract the proper bias
        if DEFAULT_BIAS_NAME in operation_attributes_dict and self.__has_bias__:
            operation_attributes_dict[DEFAULT_BIAS_NAME] = bias

        # check that the bias have parameter not empty parameter
        if not self.__has_bias__:
            operation_attributes_dict[DEFAULT_BIAS_NAME] = None

        # preform the feed-forward operations
        x = operation(x, weight, **operation_attributes_dict)

        # preform output quantization
        if self.is_quantize_output:
            self.output_initializer, self.output_range_observer = self.__pre_quantization__(
                self.output_initializer, self.output_quantizer_function, self.output_range_observer, x
            )
            x, x_q = __preform_tensor_mapping__(x, self.output_initializer, self.output_quantizer_function)

        return x

    @staticmethod
    def unwrap(obj):
        """
        Unwrap the layer
        :param obj: object to unwrap
        :return: unwrapped object
        """
        TransformationLayerQuantizerWrapper.__post_quantization__(obj)
        return obj


class GenericLayerQuantizerWrapper(LayerQuantizerWrapper):
    """
    Basic wrapper for quantization of generic tasks.
    """

    def __init__(
        self,
        torch_object: Module,
        range_observer: RangeObserver,
        initializer: QuantizationInitializer,
        quantizer_function: PyQFunction,
    ):
        super(GenericLayerQuantizerWrapper, self).__init__(
            torch_object, range_observer, initializer, quantizer_function
        )

        parameter_names = [
            name
            for name, parameter in self._wrapped_object.named_parameters()
            if not (
                DEFAULT_WRAPPED_OBJECT_NAME in name
                or DEFAULT_INITIALIZER_NAME in name
                or DOT in name
                or parameter.numel() == 1
            )
        ]

        parameter_dictionary = ParameterDict(
            {name.replace(DOT, UNDERSCORE): operator.attrgetter(name)(self._wrapped_object) for name in parameter_names}
        )

        self.parameter_names = parameter_names
        self.parameter_dictionary = parameter_dictionary
        self.set_initializers(parameter_dictionary, initializer, range_observer)
        self.initializer = None

    def set_initializers(
        self, parameter_dictionary: ParameterDict, initializer: QuantizationInitializer, range_observer: RangeObserver
    ):
        self.initializer_dictionary = ModuleDict()
        for parameter_name in parameter_dictionary:
            initializer = initializer.__class__()
            initializer.__set_parameter_range_observer__(range_observer)
            self.initializer_dictionary.update({parameter_name: initializer})

    def __count_nonzero_for_parameters__(self):
        nonzero_counter = 0
        for parameter_name in self.quantized_parameter_dictionary:
            nonzero_counter += count_nonzero(self.quantized_parameter_dictionary[parameter_name])
        return nonzero_counter

    def __count_numel_parameters__(self):
        nonzero_counter = 0
        for parameter_name in self.quantized_parameter_dictionary:
            nonzero_counter += self.quantized_parameter_dictionary[parameter_name].numel()
        return nonzero_counter

    def __pre_quantization__(self, range_observer, initializer, data):
        """
        Build the initialization  for a quantizer and register the parameters for it.
        """
        # initialization the weight quantizer if it's not initialized
        initializer.build(data)
        quantized_data = __preform_tensor_mapping__(data, initializer, Quantizer())
        range_observer.observe(quantized_data)
        initializer.set_parameter_range_observer(range_observer)
        self.register_non_persistent_buffer_per_layer(IS_QUANTIZED, True)
        self.register_non_persistent_buffer_per_layer(DEFAULT_MODEL_BIT_WIDTH_NAME, range_observer.bits)

    def __quantize__(self):
        """
        Quantize the bias of the layer based on the quantizers that provided to the constractor
        """
        self.dequantized_parameter_dictionary = ParameterDict()
        self.quantized_parameter_dictionary = ParameterDict()
        for parameter_name in self.parameter_names:
            formatted_parameter_name = parameter_name.replace(DOT, UNDERSCORE)
            non_quantized_value = operator.attrgetter(parameter_name)(self._wrapped_object)
            quantized_value = __preform_tensor_mapping__(
                non_quantized_value,
                self.initializer_dictionary[formatted_parameter_name],
                self.parameter_quantizer_function,
            )

            quantized_value = Parameter(quantized_value)
            setattr_nested(self._wrapped_object, parameter_name, quantized_value)

            self.quantized_parameter_dictionary.update({formatted_parameter_name: quantized_value})

    def __update_layer_parameters__(self):
        """
        Update the all the parameters for the layer with their quantized values, update the quantization function with
        dequantize function.
        """
        for parameter_name in self.parameter_names:
            # replace the dot with underscore to extract torch modules
            formatted_parameter_name = parameter_name.replace(DOT, UNDERSCORE)

            quantized_parameter = Parameter(self.quantized_parameter_dictionary[formatted_parameter_name])
            operator.attrgetter(parameter_name)(self._wrapped_object).data = quantized_parameter

        self.__set_parameter_quantizer_function__(IdentityFunction())

        """
        iterate over each initializer and extract its parameters to inject it inside the wrapper's scope,
        Then assign each initializer with value equals to `None`, and finally assign `initializer_dictionary` to None
        """

        for parameter_name in self.parameter_names:
            formatted_parameter_name = parameter_name.replace(DOT, UNDERSCORE)
            self.initializer_dictionary[formatted_parameter_name] = DictionaryToEmptyInitializer(
                self.initializer_dictionary[formatted_parameter_name].get_learned_parameters()
            )
            self.parameter_dictionary[formatted_parameter_name] = None

        [delattr(self, name) for name in self.state_dict().keys() if name.startswith(DEFAULT_AUXILIARY_NAME)]

    def __post_quantization__(self):
        """
        Store the quantized parameters instead of the layer parameters
        """

        # check that layer has wrapped object, if not skip unwrapping this layer
        if not hasattr(self, DEFAULT_WRAPPED_OBJECT_NAME) or not isinstance(self, LayerQuantizerWrapper):
            return

        if not hasattr(self, AUXILIARY_IS_QUANTIZED):
            warnings.warn("{} was applied as wrapper, but no quantization happen.".format(self))
            return

        if not getattr(self, AUXILIARY_IS_QUANTIZED):
            warnings.warn(
                "{} doesn't preform the feedforward, or have no parameter to quantize,try to increase "
                "num_layers.".format(self)
            )
            self = getattr(self, DEFAULT_WRAPPED_OBJECT_NAME)
            return

        self.__update_layer_parameters__()

    def __call__(self, *args, **kwargs):
        """
        Override the feedforward call to quantize the activation and the parameters and then call the layer feedforward
        with the dequantized activation and parameters to have complete differentiable quantization pipeline
        """

        # iterate over parameters to initialize the learnable parameters
        for parameter_name in self.parameter_names:
            formatted_parameter_name = parameter_name.replace(DOT, UNDERSCORE)
            parameter_values = operator.attrgetter(parameter_name)(self._wrapped_object)
            self.__pre_quantization__(
                self.parameter_range_observer, self.initializer_dictionary[formatted_parameter_name], parameter_values
            )

        # preform the quantization
        self.__quantize__()

        # count non-zero values and multiply them by the precision of the layer
        physical_size = self.__count_nonzero_for_parameters__() * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(DEFAULT_MODEL_SIZE_NAME, physical_size)

        # count values and multiply them by the precision of the layer
        full_physical_size = self.__count_numel_parameters__() * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(DEFAULT_MODEL_FULL_SIZE_NAME, full_physical_size)

        # preform the feed-forward operations
        x = super(GenericLayerQuantizerWrapper, self).__call__(*args, **kwargs)

        # pre-compute the output statistics for the output data
        self.__pre_quantization__(self.output_range_observer, self.output_initializer, x)
        x = __preform_tensor_mapping__(x, self.output_initializer, self.output_quantizer_function)
        return x

    @staticmethod
    def unwrap(obj):
        """
        Unwrap the layer
        :param obj: object to unwrap
        :return: unwrapped object
        """
        GenericLayerQuantizerWrapper.__post_quantization__(obj)
        return obj
