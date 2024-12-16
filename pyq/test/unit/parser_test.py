import functools
import os
from unittest import TestCase, main

import torch
import torchvision

from pyq.core import TorchModelParser, TorchModuleWrapper
from pyq.utils import GCN, ConvNet, CustomLinear, equal_state_dictionary


class TestWrapper(TestCase):
    def setUp(self):
        self.custom_layer = CustomLinear(in_features=10, out_features=10)
        self.wrapped_layer = TorchModuleWrapper(self.custom_layer)
        self.unwrapped_layer = self.wrapped_layer.deconstruct_wrap(self.wrapped_layer)

    def test_layer_output(self):
        random_input = torch.rand(10)
        custom_layer_output = self.custom_layer(random_input)
        wrapped_layer_output = self.wrapped_layer(random_input)
        unwrapped_layer_output = self.unwrapped_layer(random_input)

        torch.testing.assert_close(custom_layer_output, wrapped_layer_output, rtol=1, atol=0)
        torch.testing.assert_close(custom_layer_output, unwrapped_layer_output, rtol=1, atol=0)

    def test_wrapped_layer_attribute(self):
        has_is_custom_layer_attribute = hasattr(self.wrapped_layer._wrapped_object, "is_custom_layer")
        has_check_wrapping_attribute = hasattr(self.wrapped_layer, "_wrapped_object")

        self.assertTrue(has_is_custom_layer_attribute)
        self.assertTrue(has_check_wrapping_attribute)

    def test_unwrapped_layer_attribute(self):
        has_is_custom_layer_attribute = hasattr(self.unwrapped_layer, "is_custom_layer")
        has_check_wrapping_attribute = hasattr(self.unwrapped_layer, "_wrapped_object")

        self.assertTrue(has_is_custom_layer_attribute)
        self.assertTrue(not has_check_wrapping_attribute)


class TestParser(TestCase):
    def setUp(self):
        self.wrapper = TorchModuleWrapper
        self.torch_parser = TorchModelParser(callable_object=self.wrapper)

        self.models_list = [
            torchvision.models.resnet18(),
            ConvNet(1, 10),
            GCN(10, 10, 10),
        ]

        self.models_input_list = [
            torch.rand(4, 3, 128, 128),
            torch.rand(4, 1, 28, 28),
            (torch.rand(4, 10, 10), torch.randint(0, 1, (2, 100))),
        ]

        self.assert_equal = functools.partial(torch.testing.assert_close, atol=0, rtol=0)

    def test_parsing(self):
        for model in self.models_list:
            model_state_dict = model.state_dict()
            torch.save(model.state_dict(), "./" + model.__class__.__name__ + ".pth")
            model.load_state_dict(torch.load("./" + model.__class__.__name__ + ".pth"))
            parsed_model = self.torch_parser.apply(model)
            torch.save(parsed_model.state_dict(), "./" + parsed_model.__class__.__name__ + ".pth")
            parsed_model.load_state_dict(torch.load("./" + parsed_model.__class__.__name__ + ".pth"))
            parsed_state_dict = parsed_model.state_dict()
            self.assertTrue(equal_state_dictionary(model_state_dict, parsed_state_dict))

    def test_parsed_model_output(self):
        for x, model in zip(*[self.models_input_list, self.models_list]):
            parsed_model = self.torch_parser.apply(model)
            unparsed_model_output = model(*x) if type(x) is tuple else model(x)
            parsed_model_output = parsed_model(*x) if type(x) is tuple else parsed_model(x)
            self.assert_equal(unparsed_model_output, parsed_model_output)

    def test_save_and_load(self):
        for x, model in zip(*[self.models_input_list, self.models_list]):
            before_save_model_output = model(*x) if type(x) is tuple else model(x)

            parsed_model = self.torch_parser.apply(model)

            file_name = "./" + parsed_model.__class__.__name__ + ".pth"
            torch.save(parsed_model.state_dict(), file_name)
            parsed_model.load_state_dict(torch.load(file_name))

            after_load_parsed_model_output = parsed_model(*x) if type(x) is tuple else parsed_model(x)

            self.assert_equal(before_save_model_output, after_load_parsed_model_output)
            os.remove(file_name)


if __name__ == "__main__":
    main()
