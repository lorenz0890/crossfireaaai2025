import warnings
from typing import Callable, Tuple

from torch import Tensor, count_nonzero, index_select, numel
from torch.autograd import Function
from torch.nn import Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops

from pyq.core.quantization.functional import DeQuantizer, PyQFunction
from pyq.core.quantization.initializer import DictionaryToEmptyInitializer, Initializer, QuantizationInitializer
from pyq.core.quantization.observer import RangeObserver
from pyq.core.quantization.wrapper import (AUXILIARY_IS_QUANTIZED, DEFAULT_AUXILIARY_NAME, UNDERSCORE,
                                           GenericLayerQuantizerWrapper, LayerQuantizerWrapper,
                                           __preform_tensor_mapping__)
from pyq.core.wrapper import DEFAULT_WRAPPED_OBJECT_NAME
from pyq.datasets.transforms import COMPUTED_MASK_NAME
from pyq.io.model import torch_deep_copy

COMMUNICATION = "communication"

MESSAGE_METHOD_NAME = "message"
AGGREGATE_METHOD_NAME = "aggregate"
MESSAGE_AGGREGATE_METHOD_NAME = "message_and_aggregate"
UPDATE_METHOD_NAME = "update"

DEFAULT_COMMUNICATION_BIT_WIDTH_NAME = "communication_bit_width"
DEFAULT_COMMUNICATION_SIZE_NAME = "communication_size"
DEFAULT_COMMUNICATION_FULL_SIZE_NAME = "communication_full_size"


def __preform_masked_quantization__(
    value: Tensor, mask: Tensor, initializer: Initializer, quantize_function: Function
) -> Tuple[Tensor, Tensor]:
    """
    Preform a masked quantization for `values` based on boolean masked, where the True values will not quantize

    :param value: values to be quantized
    :param mask: boolean tensor that have the same shape as `values`
    :param initializer: object that contains learnable parameters that will be used in quantization
    :param quantize_function: quantization function that use to map the values from domain to different domain

    :return: dequantized and quantized values
    """

    dequantized_value, quantized_value = __preform_tensor_mapping__(value, initializer, quantize_function)

    dequantized_value[mask] = value[mask]
    dequantized_value[~mask] = dequantized_value[~mask]

    return dequantized_value, quantized_value


class CommunicationGraphQuantizerWrapper(GenericLayerQuantizerWrapper):
    """
    TODO (Samir): write the docstring for this class when the major refactor is done
    """

    def __init__(
        self,
        torch_object: Module,
        range_observer: RangeObserver,
        initializer: QuantizationInitializer,
        quantizer_function: PyQFunction,
        communication_range_observer: RangeObserver,
        communication_initializer: QuantizationInitializer,
        communication_quantizer_function: PyQFunction,
    ):
        super(CommunicationGraphQuantizerWrapper, self).__init__(
            torch_object, range_observer, initializer, quantizer_function
        )

        self.set_communication_range_observer(communication_range_observer)
        self.set_communication_quantizer_function(communication_quantizer_function)

        is_message_passing_layer = isinstance(torch_object, MessagePassing)
        if is_message_passing_layer:

            self.default_message_method = torch_deep_copy(getattr(torch_object, MESSAGE_METHOD_NAME))
            self.default_aggregate_method = torch_deep_copy(getattr(torch_object, AGGREGATE_METHOD_NAME))
            self.default_message_and_aggregate_method = torch_deep_copy(
                getattr(torch_object, MESSAGE_AGGREGATE_METHOD_NAME)
            )
            self.default_update_method = torch_deep_copy(getattr(torch_object, UPDATE_METHOD_NAME))

            setattr(torch_object, MESSAGE_METHOD_NAME, self.message)
            setattr(torch_object, AGGREGATE_METHOD_NAME, self.aggregate)
            setattr(torch_object, MESSAGE_AGGREGATE_METHOD_NAME, self.message_and_aggregate)
            setattr(torch_object, UPDATE_METHOD_NAME, self.update)

            self.set_message_initializer(communication_initializer, communication_range_observer)
            self.set_aggregate_initializer(communication_initializer, communication_range_observer)
            self.set_message_and_aggregate_initializer(communication_initializer, communication_range_observer)
            self.set_update_initializer(communication_initializer, communication_range_observer)

            self.register_non_persistent_buffer_per_layer(
                DEFAULT_COMMUNICATION_BIT_WIDTH_NAME, communication_range_observer.bits
            )

    def set_communication_quantizer_function(self, communication_quantizer_function: PyQFunction):
        """
        Setter for the quantizer function in the quantizer wrapper
        """
        self.communication_quantizer_function = communication_quantizer_function

    def set_communication_range_observer(self, communication_range_observer: RangeObserver):
        """
        Setter for the range observer in the quantizer wrapper
        """
        self.communication_range_observer = communication_range_observer.__class__(
            **communication_range_observer.get_ranger_configuration()
        ).to(self.device)

    def set_message_initializer(self, initializer: Initializer, range_observer: RangeObserver):
        """
        Setter for the initializer in the message wrapper
        """
        self.message_initializer = initializer.__class__().to(self.device)
        self.message_initializer.set_range_observer(range_observer)

    def set_aggregate_initializer(self, initializer: Initializer, range_observer: RangeObserver):
        """
        Setter for the initializer in the aggregate wrapper
        """
        self.aggregate_initializer = initializer.__class__().to(self.device)
        self.aggregate_initializer.set_range_observer(range_observer)

    def set_message_and_aggregate_initializer(self, initializer: Initializer, range_observer: RangeObserver):
        """
        Setter for the initializer in the message and aggregate wrapper
        """
        self.message_and_aggregate_initializer = initializer.__class__().to(self.device)
        self.message_and_aggregate_initializer.set_range_observer(range_observer)

    def set_update_initializer(self, initializer: Initializer, range_observer: RangeObserver):
        """
        Setter for the initializer in the update wrapper
        """
        self.update_initializer = initializer.__class__().to(self.device)
        self.update_initializer.set_range_observer(range_observer)

    def message(self, *args, **kwargs):
        """
        construct the message of node pairs (x_i, x_j)

        1. args are the output of __collect__, and kwargs in propagate. e.g x_j, edge_attr, size
        2. construct node i's messages by using variables suffixed with _i, _j,
        that’s why your see arguments with suffix _i, _j
        """
        message_output = self.default_message_method(*args, **kwargs)
        self.__pre_quantization__(self.parameter_range_observer, self.message_initializer, message_output)
        dequantized_message_output, quantized_message_output = __preform_tensor_mapping__(
            message_output, self.message_initializer, self.communication_quantizer_function
        )

        # count non-zero values and multiply them by the precision of the layer
        message_size = count_nonzero(quantized_message_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([MESSAGE_METHOD_NAME, DEFAULT_COMMUNICATION_SIZE_NAME]), message_size
        )

        # count non-zero values and multiply them by the precision of the layer
        full_message_size = numel(quantized_message_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([MESSAGE_METHOD_NAME, DEFAULT_COMMUNICATION_FULL_SIZE_NAME]), full_message_size
        )

        return dequantized_message_output

    def aggregate(self, *args, **kwargs):
        """
        aggregate message from neighbors.

        arguments: the output of message, and kwargs in propagate
        aggregate method e.g.: mean, add, max, min.
        """
        aggregation_output = self.default_aggregate_method(*args, **kwargs)
        self.__pre_quantization__(self.parameter_range_observer, self.aggregate_initializer, aggregation_output)
        dequantized_aggregation_output, quantized_aggregation_output = __preform_tensor_mapping__(
            aggregation_output, self.aggregate_initializer, self.communication_quantizer_function
        )

        # count non-zero values and multiply them by the precision of the layer
        aggregation_size = count_nonzero(quantized_aggregation_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([AGGREGATE_METHOD_NAME, DEFAULT_COMMUNICATION_SIZE_NAME]), aggregation_size
        )

        # count non-zero values and multiply them by the precision of the layer
        full_aggregation_size = numel(quantized_aggregation_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([AGGREGATE_METHOD_NAME, DEFAULT_COMMUNICATION_FULL_SIZE_NAME]), full_aggregation_size
        )

        return dequantized_aggregation_output

    def message_and_aggregate(self, *args, **kwargs):
        """
        Optional, and not used for some models.
        Combine the call of both message and aggregate
        """
        message_aggregate_output = self.default_message_and_aggregate_method(*args, **kwargs)
        self.__pre_quantization__(
            self.parameter_range_observer, self.message_and_aggregate_initializer, message_aggregate_output
        )
        (dequantized_message_aggregate_output, quantized_message_aggregate_output,) = __preform_tensor_mapping__(
            message_aggregate_output, self.message_and_aggregate_initializer, self.communication_quantizer_function
        )

        # count non-zero values and multiply them by the precision of the layer
        message_aggregate_size = count_nonzero(message_aggregate_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([MESSAGE_AGGREGATE_METHOD_NAME, DEFAULT_COMMUNICATION_SIZE_NAME]), message_aggregate_size
        )

        # count non-zero values and multiply them by the precision of the layer
        full_message_aggregate_size = numel(quantized_message_aggregate_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([MESSAGE_AGGREGATE_METHOD_NAME, DEFAULT_COMMUNICATION_FULL_SIZE_NAME]),
            full_message_aggregate_size,
        )

        return dequantized_message_aggregate_output

    def update(self, *args, **kwargs):
        """
        update embedding of Node i with aggregated message , i in V
        e.g. aggregated neighbor message and self message

        arguments: the output of aggregate, and kwargs in propagate
        """
        update_output = self.default_update_method(*args, **kwargs)
        self.__pre_quantization__(self.parameter_range_observer, self.update_initializer, update_output)
        dequantized_update_output, quantized_update_output = __preform_tensor_mapping__(
            update_output, self.update_initializer, self.communication_quantizer_function
        )

        # count non-zero values and multiply them by the precision of the layer
        update_size = count_nonzero(quantized_update_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([UPDATE_METHOD_NAME, DEFAULT_COMMUNICATION_SIZE_NAME]), update_size
        )

        # count non-zero values and multiply them by the precision of the layer
        full_update_size = numel(quantized_update_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([UPDATE_METHOD_NAME, DEFAULT_COMMUNICATION_FULL_SIZE_NAME]), full_update_size
        )

        return dequantized_update_output

    def __update_layer_parameters__(self):
        super(CommunicationGraphQuantizerWrapper, self).__update_layer_parameters__()

        self.set_communication_quantizer_function(DeQuantizer())

        # remove or replace unnecessary parameters
        non_learnable_message_initializer = DictionaryToEmptyInitializer(
            self.message_initializer.get_learned_parameters()
        )
        non_learnable_aggregate_initializer = DictionaryToEmptyInitializer(
            self.aggregate_initializer.get_learned_parameters()
        )
        non_learnable_message_and_aggregate_initializer = DictionaryToEmptyInitializer(
            self.message_and_aggregate_initializer.get_learned_parameters()
        )
        non_learnable_update_initializer = DictionaryToEmptyInitializer(
            self.update_initializer.get_learned_parameters()
        )

        self.message_initializer = non_learnable_message_initializer
        self.aggregate_initializer = non_learnable_aggregate_initializer
        self.message_and_aggregate_initializer = non_learnable_message_and_aggregate_initializer
        self.update_initializer = non_learnable_update_initializer

        [delattr(self, name) for name in self.state_dict().keys() if name.startswith(DEFAULT_AUXILIARY_NAME)]

    def __post_quantization__(self):
        # check that layer has wrapped object, if not skip unwrapping this layer
        if not isinstance(self, LayerQuantizerWrapper):
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

    @staticmethod
    def unwrap(obj):
        """
        Unwrap the layer
        :param obj: object to unwrap
        :return: unwrapped object
        """
        # check that layer has wrapped object, if not skip unwrapping this layer
        if not isinstance(obj, LayerQuantizerWrapper):
            return obj

        CommunicationGraphQuantizerWrapper.__post_quantization__(obj)
        obj = GenericLayerQuantizerWrapper.unwrap(obj)
        return obj


class SamplerCommunicationGraphQuantizerWrapper(CommunicationGraphQuantizerWrapper):
    """
    TODO (Samir): write the docstring for this class when the major refactor is done
    """

    def __init__(
        self,
        torch_object: Module,
        range_observer: RangeObserver,
        initializer: QuantizationInitializer,
        quantizer_function: Function,
        communication_range_observer: RangeObserver,
        communication_initializer: QuantizationInitializer,
        communication_quantizer_function: Function,
        communication_probability_to_mask_function: Callable,
        reconstruct_self_loops_in_communication: bool = False,
    ):
        super(SamplerCommunicationGraphQuantizerWrapper, self).__init__(
            torch_object,
            range_observer,
            initializer,
            quantizer_function,
            communication_range_observer,
            communication_initializer,
            communication_quantizer_function,
        )

        self.mask = None
        self.edge_mask = None
        self.reconstruct_self_loops = reconstruct_self_loops_in_communication
        self.probability_to_mask_function = communication_probability_to_mask_function

    def message(self, *args, **kwargs):
        if not self.training:
            return super(SamplerCommunicationGraphQuantizerWrapper, self).message(*args, **kwargs)

        message_output = self.default_message_method(*args, **kwargs)
        dequantized_message_output, quantized_message_output = __preform_masked_quantization__(
            message_output, self.edge_mask, self.message_initializer, self.communication_quantizer_function
        )

        # count non-zero values and multiply them by the precision of the layer
        message_size = count_nonzero(quantized_message_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([MESSAGE_METHOD_NAME, DEFAULT_COMMUNICATION_SIZE_NAME]), message_size
        )

        # count non-zero values and multiply them by the precision of the layer
        full_message_size = numel(quantized_message_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([MESSAGE_METHOD_NAME, DEFAULT_COMMUNICATION_FULL_SIZE_NAME]), full_message_size
        )
        return dequantized_message_output

    def aggregate(self, *args, **kwargs):
        if not self.training:
            return super(SamplerCommunicationGraphQuantizerWrapper, self).aggregate(*args, **kwargs)

        aggregate_output = self.default_aggregate_method(*args, **kwargs)
        dequantized_aggregation_output, quantized_aggregation_output = __preform_masked_quantization__(
            aggregate_output, self.mask, self.aggregate_initializer, self.communication_quantizer_function
        )

        # count non-zero values and multiply them by the precision of the layer
        aggregation_size = count_nonzero(quantized_aggregation_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([AGGREGATE_METHOD_NAME, DEFAULT_COMMUNICATION_SIZE_NAME]), aggregation_size
        )

        # count non-zero values and multiply them by the precision of the layer
        full_aggregation_size = numel(quantized_aggregation_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([AGGREGATE_METHOD_NAME, DEFAULT_COMMUNICATION_FULL_SIZE_NAME]), full_aggregation_size
        )

        return dequantized_aggregation_output

    def message_and_aggregate(self, *args, **kwargs):
        if not self.training:
            return super(SamplerCommunicationGraphQuantizerWrapper, self).message_and_aggregate(*args, **kwargs)

        message_aggregate_output = self.default_message_and_aggregate_method(*args, **kwargs)
        dequantized_message_aggregate_output, quantized_message_aggregate_output = __preform_masked_quantization__(
            message_aggregate_output,
            self.edge_mask,
            self.message_aggregate_initializer,
            self.communication_quantizer_function,
        )

        # count non-zero values and multiply them by the precision of the layer
        message_aggregate_size = count_nonzero(quantized_message_aggregate_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([MESSAGE_AGGREGATE_METHOD_NAME, DEFAULT_COMMUNICATION_SIZE_NAME]), message_aggregate_size
        )

        # count non-zero values and multiply them by the precision of the layer
        full_message_aggregate_size = numel(quantized_message_aggregate_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([MESSAGE_AGGREGATE_METHOD_NAME, DEFAULT_COMMUNICATION_FULL_SIZE_NAME]),
            full_message_aggregate_size,
        )

        return dequantized_message_aggregate_output

    def update(self, *args, **kwargs):
        if not self.training:
            return super(SamplerCommunicationGraphQuantizerWrapper, self).update(*args, **kwargs)

        update_output = self.default_update_method(*args, **kwargs)
        dequantized_update_output, quantized_update_output = __preform_masked_quantization__(
            update_output, self.mask, self.update_initializer, self.communication_quantizer_function
        )

        # count non-zero values and multiply them by the precision of the layer
        update_size = count_nonzero(quantized_update_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([UPDATE_METHOD_NAME, DEFAULT_COMMUNICATION_SIZE_NAME]), update_size
        )

        # count non-zero values and multiply them by the precision of the layer
        full_update_size = numel(quantized_update_output) * self.parameter_range_observer.bits
        self.register_non_persistent_buffer_per_layer(
            UNDERSCORE.join([UPDATE_METHOD_NAME, DEFAULT_COMMUNICATION_FULL_SIZE_NAME]), full_update_size
        )

        return dequantized_update_output

    def __call__(self, *args, **kwargs):
        assert len(args) >= 2 and args[1].shape[0] == 2, "the second argument is not matching to `edge_index`."
        assert COMPUTED_MASK_NAME in kwargs or not self.training, (
            "{} is not provided, use the transform `MaskMaker` or `ProbabilityMaskMaker` then inject the argument "
            "`computed_mask` using the `model_editor`".format(COMPUTED_MASK_NAME)
        )

        if self.training:
            x = args[0]
            edge_index = args[1]
            if self.reconstruct_self_loops:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            prob_mask = kwargs.pop(COMPUTED_MASK_NAME)
            # update the masks
            self.mask = self.probability_to_mask_function(prob_mask)
            self.edge_mask = index_select(self.mask, 0, edge_index[0])

        if not self.training and COMPUTED_MASK_NAME in kwargs:
            kwargs.pop(COMPUTED_MASK_NAME)

        return super(SamplerCommunicationGraphQuantizerWrapper, self).__call__(*args, **kwargs)
