import inspect
from typing import List, Optional

from torch.nn import Module

from pyq.experiment.function_builder import (DEFAULT_FORWARD_METHOD_NAME, BindStringToObjectFunction,
                                             StringFunctionArgumentUpdater, StringFunctionRedefiner)

DOT = "."


class TorchModelEditor:
    def __init__(self, torch_model: Module):
        self.set_torch_model(torch_model)
        self.forward_function_as_reference = getattr(torch_model, DEFAULT_FORWARD_METHOD_NAME)
        self.forward_function_as_sting = inspect.getsource(self.forward_function_as_reference)

    def set_torch_model(self, torch_model):
        """
        set the value of torch model
        :param torch_model: an instance of torch model
        """
        self.torch_model = torch_model

    def get_torch_model(self):
        """
        get the value of torch model
        :return instance_object: an instance of torch model
        """
        return self.torch_model

    def set_forward_function_as_sting(self, forward_function_as_sting):
        """
        set the forward function as string value
        :param forward_function_as_sting: an instance of forward function
        """
        self.forward_function_as_sting = forward_function_as_sting

    def get_forward_function_as_sting(self):
        """
        get the value for the forward function as string
        :return : forwarded function in a string format
        """
        return self.forward_function_as_sting

    def __bind_forward_function_to_model__(self, function_as_string: str):
        """
        set the new forward function to the `torch_model`
        :param function_as_string: a forward function as string
        """
        self.set_forward_function_as_sting(function_as_string)
        function_binder = BindStringToObjectFunction(function_as_string=function_as_string)
        # TODO (Samir): remove the redundant for setting code function
        function_binder.set_coded_function_as_string(function_as_string)
        updated_model = function_binder.bind_function_inside_instance(self.torch_model)
        self.set_torch_model(updated_model)

    def update_arguments_for_forward_function(
        self, argument_names_to_be_inserted_in_forward, instance_names_as_regex_to_be_updated_in_forward
    ):
        """
        Update the argument that accepted in the forward function, and pass the new arguments to some instances

        :param argument_names_to_be_inserted_in_forward: new argument names to be added to the forward function
        :param instance_names_as_regex_to_be_updated_in_forward: instances that will use the new added arguments
        """
        assert not any(
            [
                argument_name in [*inspect.signature(self.forward_function_as_reference).parameters]
                for argument_name in argument_names_to_be_inserted_in_forward
            ]
        ), "{} one of these arguments is already in function definition header".format(
            argument_names_to_be_inserted_in_forward
        )

        function_string_builder = StringFunctionArgumentUpdater(function_as_string=self.get_forward_function_as_sting())

        function_string_builder.update_arguments_in_function_header(argument_names_to_be_inserted_in_forward, False)
        function_string_builder.update_arguments_passed_to_instances(instance_names_as_regex_to_be_updated_in_forward)
        updated_function_as_string = function_string_builder.get_coded_function_as_string()

        self.__bind_forward_function_to_model__(updated_function_as_string)

    def insert_code_in_forward(
        self, lines_of_code_to_be_inserted_inside_forward: List[str], insert_at: Optional[str] = "begin"
    ):
        """
        insert a line of code in the forward function

        :param lines_of_code_to_be_inserted_inside_forward: line of code to be inserted in the forward function
        :param insert_at: string to indicate whether if the line of coed will be inserted in the beginning or at the end
        """
        assert insert_at in ["begin", "end"], "either `insert_at_begin` or `insert_at_end` should be provided."

        redefiner_functions = StringFunctionRedefiner(function_as_string=self.get_forward_function_as_sting())
        redefiner_functions.set_coded_function_as_string(self.get_forward_function_as_sting())
        if insert_at == "begin":
            for code_as_string in lines_of_code_to_be_inserted_inside_forward:
                redefiner_functions.add_line_of_code_at_begin(code_as_string)
        if insert_at == "end":
            for code_as_string in lines_of_code_to_be_inserted_inside_forward:
                redefiner_functions.add_line_of_code_at_end(code_as_string)
        updated_function_as_string = redefiner_functions.get_coded_function_as_string()

        self.__bind_forward_function_to_model__(updated_function_as_string)

    def insert_code_in_forward_scope(
        self, lines_of_code_to_be_inserted_inside_forward_scope: List[str], insert_at: Optional[str] = "begin"
    ):
        """
        insert a line of code in the forward function

        :param lines_of_code_to_be_inserted_inside_forward: line of code to be inserted in the forward function
        :param insert_at: string to indicate whether if the line of coed will be inserted in the beginning or at the end
        """
        redefiner_functions = StringFunctionRedefiner(function_as_string=self.get_forward_function_as_sting())
        for code_as_string in lines_of_code_to_be_inserted_inside_forward_scope:
            redefiner_functions.add_line_of_code_in_header_scope(code_as_string)
        updated_function_as_string = redefiner_functions.get_coded_function_as_string()

        self.__bind_forward_function_to_model__(updated_function_as_string)


def setattr_nested(obj, path, value):
    """
    Accept a dotted path to a nested attribute to set.
    """
    path, _, target = path.rpartition(DOT)
    if len(path):
        for attr_name in path.split(DOT):
            obj = getattr(obj, attr_name)
    setattr(obj, target, value)
