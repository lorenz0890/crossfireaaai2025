import re
from abc import ABC
from collections import OrderedDict
from operator import attrgetter
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from torch.nn import Identity, Module
from torch_geometric.nn import MessagePassing

from pyq.core.quantization.communication_wrapper import (COMMUNICATION, CommunicationGraphQuantizerWrapper,
                                                         SamplerCommunicationGraphQuantizerWrapper)
from pyq.core.quantization.supported_operations import layer_to_operation
from pyq.core.quantization.wrapper import (DEFAULT_BIAS_NAME, GenericLayerQuantizerWrapper, LayerQuantizerWrapper,
                                           TransformationLayerQuantizerWrapper)
from pyq.core.wrapper import DEFAULT_WRAPPED_OBJECT_NAME
from pyq.models.editor import DOT
from pyq.models.wrapper import TorchModelWrapper

BASIC_MESSAGE_PASSING_WRAPPER = [
    GenericLayerQuantizerWrapper,
    CommunicationGraphQuantizerWrapper,
    SamplerCommunicationGraphQuantizerWrapper,
]

BASIC_MESSAGE_PASSING_UNWRAPPER = [
    GenericLayerQuantizerWrapper.unwrap,
    CommunicationGraphQuantizerWrapper.unwrap,
    SamplerCommunicationGraphQuantizerWrapper.unwrap,
]


def traverse_dict(dictionary: dict, path: Optional[Union[List, None]] = None):
    """
    :param dictionary: dictionary to traverse the keys in it and have it in a list form
    :param path: optional list to allow recursively call to extract the nested keys
    :return: generator to yield the list of nested keys of each key in the dictionary
    """
    if not path:
        path = []
    if isinstance(dictionary, dict):
        for x in dictionary.keys():
            local_path = path[:]
            local_path.append(x)
            for b in traverse_dict(dictionary[x], local_path):
                yield b
    else:
        yield path, dictionary


def __add_indent__(s_, number_of_spaces):
    # modified version of the torch source code
    # source: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py#L29

    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(number_of_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def __extract_variables_names_from_module__(model: Module):
    # modified version of the torch source code
    # source: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py#L1883
    main_str = ""
    extra_lines = []
    child_lines = []
    for key, module in model._modules.items():
        if module is None:
            continue
        mod_str = ": {" + __extract_variables_names_from_module__(module) + "}" if len(list(module.children())) else ""
        mod_str = __add_indent__(mod_str, 2)
        child_lines.append("'" + key + "'" + mod_str)
    lines = extra_lines + child_lines
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += "\n" + "\n".join(lines)
    return main_str


def produce_sorted_layer_path_from_repr_(model: Module):
    module_name = __extract_variables_names_from_module__(model)
    module_name_parsed = (
        "{"
        + module_name.replace("'\n", "': None,\n").replace("'}", "': None,}").replace("}\n", "},\n")
        + ("}" if module_name[-1] == "}" else ": None}")
    )
    module_dict = eval(module_name_parsed)
    module_list_with_nones = traverse_dict(module_dict)
    layers_path = [DOT.join(name[0]) for name in module_list_with_nones]
    return layers_path


def produce_layers_path(model: Module):
    """
    Independent function that extracts the parameters from the torch model by accessing its `state_dict`.

    :param model: torch model that could have complex/nested layers in its architecture
    :return: list of the complete names of the operation layers without redundancy
    """
    layers_path = produce_sorted_layer_path_from_repr_(model)
    state_dict_keys = list(model.state_dict().keys()) + [*dict(model.named_modules()).keys()]
    layers_path = list(OrderedDict.fromkeys(layers_path + [DOT.join(k.split(DOT)[:-1]) for k in state_dict_keys]))

    layers_path.sort(key=len)
    # ** It's importance to sort the layer in the reversed based on the length of the string to wrap the children
    # before their parents **
    return list(filter(None, layers_path[::-1]))


class BaseModelParser(ABC):
    def __init__(
        self,
        callable_object: Union[Callable, Type],
        callable_object_kwargs: Optional[Dict[str, Any]] = {},
        callable_object_for_nonparametric: Optional[Union[Callable, Type]] = lambda args: args,
        callable_object_for_nonparametric_kwargs: Optional[Dict[str, Any]] = {},
        skip_parsing: Optional[bool] = False,
    ):
        """
        Model parser is an abstract class to iterate through the model components and apply an operation on these
        components, add, or remove them.

        :param callable_object: Class/function that going to wrap/apply on the model components/operation
        :param callable_object_for_nonparametric: Class/function that going to wrap/apply on the functions
        :param callable_object_kwargs: Arguments that will be passed to the class instructor or callable function
        :param callable_object_for_nonparametric_kwargs: Arguments that will be passed to the callable function
        """
        self.default_callable_object = callable_object
        self.default_callable_object_kwargs = callable_object_kwargs
        self.default_callable_object_for_nonparametric = callable_object_for_nonparametric
        self.default_callable_object_for_nonparametric_kwargs = callable_object_for_nonparametric_kwargs
        self.skip_parsing = skip_parsing
        self.__reset_callable_object__()

    def __reset_callable_object__(self):
        self.callable_object = self.default_callable_object
        self.callable_object_kwargs = self.default_callable_object_kwargs
        self.callable_object_for_nonparametric = self.default_callable_object_for_nonparametric
        self.callable_object_for_nonparametric_kwargs = self.default_callable_object_for_nonparametric_kwargs

    def __set_callable_object__(
        self,
        callable_object: Union[Callable, Type],
        callable_object_kwargs: Optional[Dict[str, Any]] = {},
        callable_object_for_nonparametric: Union[Callable, Type] = lambda args: args,
        callable_object_for_nonparametric_kwargs: Optional[Dict[str, Any]] = {},
    ):
        self.callable_object = callable_object
        self.callable_object_kwargs = callable_object_kwargs
        self.callable_object_for_nonparametric = callable_object_for_nonparametric
        self.callable_object_for_nonparametric_kwargs = callable_object_for_nonparametric_kwargs

    def __get_callable_object__(self):
        return (
            self.callable_object,
            self.callable_object_kwargs.copy(),
            self.callable_object_for_nonparametric,
            self.default_callable_object_for_nonparametric_kwargs.copy(),
        )

    def apply(self, torch_model: Module):
        """
        Travers through the model components/operation and use the `callable_object` to perform operations on the
        components/operation

        :param torch_model: Model that is going to traverse through
        :return: Instance of the model after traversing and applying the `callable_class`
        """
        raise NotImplementedError(f"{self.__class__.__name__} suppose to be abstract class")


class TorchModelParser(BaseModelParser):
    def __init__(
        self,
        callable_object: Union[Callable, Type],
        callable_object_kwargs: Optional[Dict[str, Any]] = {},
        callable_object_for_nonparametric: Union[Callable, Type] = lambda args: args,
        callable_object_for_nonparametric_kwargs: Optional[Dict[str, Any]] = {},
        remove_layers_bias: Optional[bool] = False,
        skip_layer_by_type: Optional[List[Type]] = (),
        skip_layer_by_regex: Optional[List[str]] = (),
        delete_layer_by_type: Optional[Tuple[Type]] = (),
        skip_parsing: Optional[bool] = False,
    ):
        super().__init__(
            callable_object,
            callable_object_kwargs,
            callable_object_for_nonparametric,
            callable_object_for_nonparametric_kwargs,
            skip_parsing,
        )
        self.remove_layers_bias = remove_layers_bias
        self.skip_layer_by_type = skip_layer_by_type
        self.skip_layer_by_regex = skip_layer_by_regex
        self.delete_layer_by_type = delete_layer_by_type

    def __apply_callable_object_on_layer__(self, variable: Module, attribute: Union[int, str]):
        """
        Store the parameters of the layer and then apply the `callable_object` on the `variable.attr` and store the
        returned value from the `callable_object` in the same location `variable.attr`.

        :param variable: layer/operation that has the string/integer `attr` in its attribute or index
        :param attribute: string or integer that could be either attribute or index for the variable
        :return: True if the assigning happened correctly otherwise, it will raise an exception
        """

        # choose the method to extract and assign the variables because it sometime could be (Sequential, ModuleList)
        # and for that same layers are stacked in a list
        __set__, __get__ = ("__setitem__", "__getitem__") if type(attribute) is int else ("__setattr__", "__getattr__")
        # getattr(variable, __set__)(attribute, ...)    equivalent to   `model.layer_i = ...`
        # getattr(variable, __get__)(attribute)         equivalent to   `model.layer_i`
        # the following line                            equivalent to   `model.layer_i = callable_object(model.layer_i)`
        layer = getattr(variable, __get__)(attribute)

        # if the layer instance of the layers to delete replace it with identity
        if isinstance(layer, tuple(self.delete_layer_by_type)):
            getattr(variable, __set__)(attribute, Identity())
            return True

        # if the layer instance of the layers to skip return true
        if isinstance(layer, tuple(self.skip_layer_by_type)) or layer is None:
            return True

        # remove the bias from the layer if it exists
        if self.remove_layers_bias and hasattr(layer, DEFAULT_BIAS_NAME):
            setattr(layer, DEFAULT_BIAS_NAME, None)

        # avoid wrapping if the layer has no parameters and the callable_object_for_nonparametric is None
        if len(layer.state_dict()) == 0 and self.callable_object_for_nonparametric is None:
            return True

        # if the layer or the wrapped object has no parameters, apply the non-parametric wrapper (activation wrapper)
        if len(layer.state_dict()) == 0 or (
            hasattr(layer, DEFAULT_WRAPPED_OBJECT_NAME) and len(layer._wrapped_object.state_dict()) == 0
        ):
            getattr(variable, __set__)(
                attribute,
                self.callable_object_for_nonparametric(layer, **self.callable_object_for_nonparametric_kwargs),
            )
            return True

        # if the layer is not instance of graph layers, and the wrapper is quantizer for graph task swap the wrapper
        if not isinstance(layer, MessagePassing) and (
            self.callable_object in BASIC_MESSAGE_PASSING_WRAPPER
            or self.callable_object in BASIC_MESSAGE_PASSING_UNWRAPPER
        ):
            callable_object, callable_object_kwargs, _, _ = self.__get_callable_object__()
            # update the `callable_object` to be normal transformation layer
            if self.callable_object in BASIC_MESSAGE_PASSING_WRAPPER:
                callable_object = TransformationLayerQuantizerWrapper
                callable_object_kwargs = {
                    key: callable_object_kwargs[key] for key in callable_object_kwargs if COMMUNICATION not in key
                }
            if (
                self.callable_object in BASIC_MESSAGE_PASSING_UNWRAPPER
                and hasattr(layer, DEFAULT_WRAPPED_OBJECT_NAME)
                and isinstance(layer._wrapped_object, tuple(layer_to_operation.keys()))
            ):
                callable_object = TransformationLayerQuantizerWrapper.unwrap
            getattr(variable, __set__)(attribute, callable_object(layer, **callable_object_kwargs))

            # reset the callable object and its argument s to the default values
            self.__reset_callable_object__()
            return True

        getattr(variable, __set__)(attribute, self.callable_object(layer, **self.callable_object_kwargs))
        return True

    def __traverse_over_layers__(self, block, names):
        """
        Recursively traverse over the `block` to access the `list_names` inside this block and then,
        when that list of names finishes, apply the `callable_object` over the leaf block.

        :param block: block that has attributes and for each attribute, there are other attributes inside it.
        :param names: list of attributes names that can be found inside the `block`.
        :return: true if the `callable_object` applied correctly otherwise, it will raise an exception
        """

        # check if no names in the `names`, then return `True`
        if len(names) == 0:
            return True

        if not hasattr(block, names[0]):
            return True

        layer_i = block.__getattr__(names[0])

        # If this is the last name then, this is the target layer/operation that `callable_object` can be applied on.
        if len(names) == 1:
            return self.__apply_callable_object_on_layer__(block, names[0])

        # if the block has `_wrapped_object` as last element apply `callable_function` on it
        if len(names) == 2 and names[1] == DEFAULT_WRAPPED_OBJECT_NAME:
            return self.__apply_callable_object_on_layer__(block, names[0])

        # extract the layers from sequential layers
        if type(names[0]) is int:
            self.__traverse_over_layers__(block[names[0]], names[1:])

        # traverse recursively in each layer
        elif any(hasattr(layer_i, attr) for attr in names[1:]):
            self.__traverse_over_layers__(layer_i, names[1:])

    def apply(self, torch_model: Module):
        """
        Iterate over `torch_model` layers to apply the callable function or wrap them using `callable_object`.

        :param torch_model: target torch model that is going to perform the operation on it
        :return: new torch Module after applying the `callable_object` on its layers
        """
        if self.skip_parsing:
            return torch_model

        parsed_model = torch_model
        layers_path_name = produce_layers_path(parsed_model)
        for layer_path in layers_path_name:
            # if the layer has the same path as the name that provided in the skip list skip parsing
            if any([re.search(regex, layer_path) for regex in self.skip_layer_by_regex]):
                continue
            split_layer_path = layer_path.split(DOT)
            self.__traverse_over_layers__(parsed_model, split_layer_path)

        return TorchModelWrapper(parsed_model)
