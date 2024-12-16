from torch.nn import Module

from pyq.core.quantization.communication_wrapper import (DEFAULT_COMMUNICATION_BIT_WIDTH_NAME,
                                                         DEFAULT_COMMUNICATION_FULL_SIZE_NAME,
                                                         DEFAULT_COMMUNICATION_SIZE_NAME)
from pyq.core.quantization.wrapper import (DEFAULT_MODEL_BIT_WIDTH_NAME, DEFAULT_MODEL_FULL_SIZE_NAME,
                                           DEFAULT_MODEL_SIZE_NAME)
from pyq.core.wrapper import DEFAULT_WRAPPED_OBJECT_NAME, TorchModuleWrapper


class TorchModelWrapper(TorchModuleWrapper):
    """
    Wrapper that wrap the torch model to compute callbacks over buffer parameters per layer.
    """

    def __init__(self, torch_model: Module):
        super().__init__(torch_model)
        self.callback_dictionary = {}

    def get_torch_model(self) -> Module:
        """
        Get the wrapped model to avoid access the model from the wrapper directly
        :return: the torch wrapped model
        """
        return getattr(self, DEFAULT_WRAPPED_OBJECT_NAME)

    def get_callback_dictionary(self) -> dict:
        """
        Get the callback dictionary to collect information about the whole model
        :return: dictionary holding names of metric for layer in keys and the metric value in it's value
        """
        return self.callback_dictionary

    def update_average_bit_widths(self):
        """
        Compute the average bit width for the whole model based on the bit width on each layer, and store the updated
        value in the `callback_dictionary`
        """
        torch_model = self.get_torch_model()
        bit_widths = [value for name, value in torch_model.named_buffers() if DEFAULT_MODEL_BIT_WIDTH_NAME in name]
        if len(bit_widths) > 0:
            self.callback_dictionary.update({DEFAULT_MODEL_BIT_WIDTH_NAME: sum(bit_widths) / len(bit_widths)})

    def update_communication_average_bit_widths(self):
        """
        Compute the average bit width for the whole model based on the bit width on each layer, and store the updated
        value in the `callback_dictionary`
        """
        torch_model = self.get_torch_model()
        bit_widths = [
            value for name, value in torch_model.named_buffers() if DEFAULT_COMMUNICATION_BIT_WIDTH_NAME in name
        ]
        if len(bit_widths) > 0:
            self.callback_dictionary.update({DEFAULT_COMMUNICATION_BIT_WIDTH_NAME: sum(bit_widths) / len(bit_widths)})

    def update_full_physical_size(self):
        """
        Compute the non-zero multiply by the bit width for the whole model based on the  non-zero elements bit width on
        each layer, and store the updated value in the `callback_dictionary`
        """
        torch_model = self.get_torch_model()
        full_physical_size = [
            value for name, value in torch_model.named_buffers() if DEFAULT_MODEL_FULL_SIZE_NAME in name
        ]
        if len(full_physical_size) > 0:
            self.callback_dictionary.update({DEFAULT_MODEL_FULL_SIZE_NAME: sum(full_physical_size)})

    def update_physical_size(self):
        """
        Compute the non-zero elements multiply by the bit width for the whole model based on the  non-zero elements
        bit width on each layer, and store the updated value in the `callback_dictionary`
        """
        torch_model = self.get_torch_model()
        physical_size = [value for name, value in torch_model.named_buffers() if DEFAULT_MODEL_SIZE_NAME in name]
        if len(physical_size) > 0:
            self.callback_dictionary.update({DEFAULT_MODEL_SIZE_NAME: sum(physical_size)})

    def update_communication_full_size(self):
        """
        Compute the number of elements multiply by the bit width for the whole model based on the  non-zero elements
        bit width on each layer, and store the updated value in the `callback_dictionary`
        """
        torch_model = self.get_torch_model()
        communication_size = [
            value for name, value in torch_model.named_buffers() if DEFAULT_COMMUNICATION_FULL_SIZE_NAME in name
        ]
        if len(communication_size) > 0:
            self.callback_dictionary.update({DEFAULT_COMMUNICATION_FULL_SIZE_NAME: sum(communication_size)})

    def update_communication_size(self):
        """
        Compute the non-zero multiply by the bit width for the whole model based on the  non-zero elements bit width on
        each layer, and store the updated value in the `callback_dictionary`
        """
        torch_model = self.get_torch_model()
        communication_size = [
            value for name, value in torch_model.named_buffers() if DEFAULT_COMMUNICATION_SIZE_NAME in name
        ]
        if len(communication_size) > 0:
            self.callback_dictionary.update({DEFAULT_COMMUNICATION_SIZE_NAME: sum(communication_size)})

    def update_all_callbacks(self):
        """
        call all the updaters to update the `callback_dictionary` based on the current state.
        """
        self.update_average_bit_widths()
        self.update_communication_average_bit_widths()
        self.update_full_physical_size()
        self.update_physical_size()
        self.update_communication_full_size()
        self.update_communication_size()
        pass


class InputQuantizerOutputDequantizerTorchModelWrapper(TorchModelWrapper):
    def __init__(self, torch_model: Module):
        super().__init__(torch_model)
