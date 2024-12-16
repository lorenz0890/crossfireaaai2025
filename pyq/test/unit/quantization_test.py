from typing import Any
from unittest import TestCase, main

import pprint
import torch
from torch.nn import Conv2d
from torch_geometric.nn import GCN

from pyq.core.quantization.functional import STEQuantizeFunction
from pyq.core.quantization.initializer import RandomInitializer, DEFAULT_INITIALIZER_NAME
from pyq.core.quantization.observer import UniformRangeObserver
from pyq.core.quantization.wrapper import (DEFAULT_BIAS_NAME, DEFAULT_WEIGHT_NAME, GenericLayerQuantizerWrapper,
                                           TransformationLayerQuantizerWrapper, IS_QUANTIZED, DEFAULT_AUXILIARY_NAME)

CONSTANTS_TO_REMOVE = [IS_QUANTIZED, DEFAULT_INITIALIZER_NAME, DEFAULT_AUXILIARY_NAME, "scale", "offset", "threshold", "quantized_parameter_dictionary"]


def remove_keys_from_dictionary(dictionary: dict, lists_of_names_skip: list):
    return {k: dictionary[k] for k in dictionary if not any([skip_name in k for skip_name in lists_of_names_skip])}


def remove_parameters_from_object(obj: Any, parameters_name: list):
    _ = [setattr(obj, parameter_name, None) for parameter_name in parameters_name]
    return obj


class TestIntegerQuantizedModels(TestCase):
    def setUp(self):
        bits = 8

        self.initializer = RandomInitializer()
        self.quantization_function = STEQuantizeFunction()
        self.range_observer = UniformRangeObserver(bits)

        self.data_point = torch.rand(1, 3, 64, 64)
        self.dataset_num_features = 3
        self.dataset_num_classes = 10
        self.kernel_size = 3

        self.graph_data_point = (torch.rand(20, 10), torch.randint(0, 1, (2, 100)))
        self.graph_dataset_num_features = 10
        self.graph_dataset_num_classes = 5
        self.num_layers = 2

    def test_quantized_convolution_integers(self):
        convolution = Conv2d(
            in_channels=self.dataset_num_features,
            out_channels=self.dataset_num_features,
            kernel_size=self.kernel_size,
        )

        quantized_convolution = TransformationLayerQuantizerWrapper(
            convolution, self.range_observer, self.initializer, self.quantization_function
        )

        quantized_convolution(self.data_point)

        quantized_convolution = TransformationLayerQuantizerWrapper.unwrap(quantized_convolution)

        quantized_convolution_state_dict = quantized_convolution.state_dict()
        for k in quantized_convolution_state_dict:
            if DEFAULT_WEIGHT_NAME in k or DEFAULT_BIAS_NAME in k:
                weight = quantized_convolution_state_dict[k]
                self.assertTrue(torch.all(weight.type(dtype=torch.int) == weight))

    def test_quantized_gcn_integers(self):
        gcn = GCN(
            in_channels=self.graph_dataset_num_features,
            hidden_channels=self.graph_dataset_num_classes,
            num_layers=self.num_layers,
        )

        for i in range(self.num_layers):
            gcn.convs[i].lin  = TransformationLayerQuantizerWrapper(gcn.convs[i].lin, self.range_observer, self.initializer, self.quantization_function)
        quantized_gcn = GenericLayerQuantizerWrapper(
            gcn, self.range_observer, self.initializer, self.quantization_function
        )

        quantized_gcn(*self.graph_data_point)

        for i in range(self.num_layers):
            gcn.convs[i].lin = TransformationLayerQuantizerWrapper.unwrap(gcn.convs[i].lin)
        quantized_gcn = GenericLayerQuantizerWrapper.unwrap(quantized_gcn)

        quantized_gat_state_dict = quantized_gcn.state_dict()
        for k in quantized_gat_state_dict:
            if k.endswith(DEFAULT_WEIGHT_NAME) or k.endswith(DEFAULT_BIAS_NAME):
                weight = quantized_gat_state_dict[k]
                self.assertTrue(torch.all(weight.type(dtype=torch.int) == weight))


class TestQuantizedConvolutionModel(TestCase):
    def setUp(self):
        bits = 8
        self.loss = torch.nn.MSELoss()

        self.initializer = RandomInitializer()
        self.quantization_function = STEQuantizeFunction()
        self.range_observer = UniformRangeObserver(bits)

        self.data_point = torch.rand(1, 3, 64, 64)
        self.dataset_num_features = 3
        self.dataset_num_classes = 10
        self.kernel_size = 3

    def test_quantized_convolution_forward(self):
        convolution = Conv2d(
            in_channels=self.dataset_num_features,
            out_channels=self.dataset_num_features,
            kernel_size=self.kernel_size,
        )
        quantized_convolution = TransformationLayerQuantizerWrapper(
            convolution, self.range_observer, self.initializer, self.quantization_function
        )
        y = quantized_convolution(self.data_point)
        self.assertTrue(y is not None)

    def test_quantized_convolution_backward(self):
        convolution = Conv2d(
            in_channels=self.dataset_num_features,
            out_channels=self.dataset_num_features,
            kernel_size=self.kernel_size,
        )
        quantized_convolution = TransformationLayerQuantizerWrapper(
            convolution, self.range_observer, self.initializer, self.quantization_function
        )
        y = quantized_convolution(self.data_point)
        loss = self.loss(y, torch.rand_like(y))
        loss.backward()
        quantized_convolution = remove_parameters_from_object(quantized_convolution, CONSTANTS_TO_REMOVE)
        params = [*quantized_convolution.parameters()]
        print({n: p.grad is not None for n, p in quantized_convolution.named_parameters()})
        self.assertTrue(all(param.grad is not None for param in params))

    def test_quantized_convolution_state_dict(self):
        convolution = Conv2d(
            in_channels=self.dataset_num_features,
            out_channels=self.dataset_num_features,
            kernel_size=self.kernel_size,
        )
        convolution_state_dict = convolution.state_dict()
        quantized_convolution = TransformationLayerQuantizerWrapper(
            convolution, self.range_observer, self.initializer, self.quantization_function
        )
        quantized_convolution(self.data_point)
        quantized_convolution = TransformationLayerQuantizerWrapper.unwrap(quantized_convolution)
        quantized_convolution_state_dict = quantized_convolution.state_dict()
        quantized_convolution_state_dict = remove_keys_from_dictionary(
            quantized_convolution_state_dict, CONSTANTS_TO_REMOVE
        )
        self.assertTrue(len(convolution_state_dict) == len(quantized_convolution_state_dict.keys()))


class TestQuantizedGraphModel(TestCase):
    def setUp(self):
        bits = 8
        self.loss = torch.nn.MSELoss()

        self.initializer = RandomInitializer()
        self.quantization_function = STEQuantizeFunction()
        self.range_observer = UniformRangeObserver(bits)

        self.graph_data_point = (torch.rand(20, 10), torch.randint(0, 1, (2, 100)))
        self.graph_dataset_num_features = 10
        self.graph_dataset_num_classes = 5
        self.num_layers = 2

    def test_gcn_quantized_forward(self):
        gcn = GCN(
            in_channels=self.graph_dataset_num_features,
            hidden_channels=self.graph_dataset_num_classes,
            num_layers=self.num_layers,
        )
        quantized_gcn = GenericLayerQuantizerWrapper(
            gcn, self.range_observer, self.initializer, self.quantization_function
        )
        y = quantized_gcn(*self.graph_data_point)
        self.assertTrue(y is not None)

    def test_gcn_quantized_backward(self):
        gcn = GCN(
            in_channels=self.graph_dataset_num_features,
            hidden_channels=self.graph_dataset_num_classes,
            num_layers=self.num_layers,
        )
        quantized_gcn = GenericLayerQuantizerWrapper(
            gcn, self.range_observer, self.initializer, self.quantization_function
        )
        y = quantized_gcn(*self.graph_data_point)
        loss = self.loss(y, torch.rand_like(y))
        loss.backward()
        quantized_gcn.__post_quantization__()
        quantized_gcn = remove_parameters_from_object(quantized_gcn, CONSTANTS_TO_REMOVE)
        params = [*quantized_gcn.parameters()]
        pprint.pprint({n: p.grad is not None for n, p in quantized_gcn.named_parameters()})
        self.assertTrue(all(param.grad is not None for param in params))

    def test_gcn_quantized_state_dict(self):
        gcn = GCN(
            in_channels=self.graph_dataset_num_features,
            hidden_channels=self.graph_dataset_num_classes,
            num_layers=self.num_layers,
        )
        gcn_state_dict = gcn.state_dict()
        quantized_gcn = GenericLayerQuantizerWrapper(
            gcn, self.range_observer, self.initializer, self.quantization_function
        )
        quantized_gcn(*self.graph_data_point)
        quantized_gcn = GenericLayerQuantizerWrapper.unwrap(quantized_gcn)
        quantized_gcn_state_dict = quantized_gcn.state_dict()
        quantized_gcn_state_dict = remove_keys_from_dictionary(quantized_gcn_state_dict, CONSTANTS_TO_REMOVE)
        self.assertTrue(len(gcn_state_dict.keys()) == len(quantized_gcn_state_dict.keys()))


if __name__ == "__main__":
    main()
