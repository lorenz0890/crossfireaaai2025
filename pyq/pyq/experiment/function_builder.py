from inspect import _empty, getsource, signature
from re import findall, sub
from typing import Any, Callable, Dict, List, Optional, Union

import black
from torch import Tensor

DEFAULT_FORWARD_METHOD_NAME = "forward"


def get_function_argument_names_and_values(instance, func: str = "__init__", get_default_values: bool = True) -> dict:
    """
    extract the names of the parameters that needed to pass for a specific function, by default it will return the
    names of the parameters for the contractor

    :param instance: instance or the object that needed to extract its parameter's names
    :param func: name of the target function as string to get its parameter's names
    :param get_default_values: boolean indicate weather to return the defaults values in the function definition or not.
    :return: dictionary holding names of parameters in its keys and the value of the attributes which have the names
             inside the instance equal to this parameter name
    """
    func_signature = signature(getattr(instance, func))
    parameter_dict = {}
    for arg_name, arg_value in func_signature.parameters.items():
        if get_default_values:
            default_value = getattr(instance, arg_name) if hasattr(instance, arg_name) else None
            parameter_dict.update({arg_name: default_value})
        elif arg_value.default is _empty:
            parameter_dict.update({arg_name: None})
    return parameter_dict


def __intersection_between_function_argument_names_and_instance_variables__(
    function: Callable, instance: Any, names_to_skip: List[str]
) -> Dict[str, Union[Union[int, float], Tensor]]:
    """
    Find the intersection between function arguments and the arguments inside instance arguments and return the values
    of these intersections from the instance.

    :param function: function to extract its arguments
    :param instance: an object to extract values from
    :param names_to_skip: names to skip during extraction
    :return: dictionary that holds the arguments as keys and the values from instance as values
    """
    arguments_dict = get_function_argument_names_and_values(function, DEFAULT_FORWARD_METHOD_NAME)
    [arguments_dict.pop(name) for name in names_to_skip if name in names_to_skip and name in arguments_dict]
    # extract values
    arguments = {name: getattr(instance, name) for name in arguments_dict.keys() if hasattr(instance, name)}
    return arguments


DEFINE_NAME = "def "
FUNCTION_NAME_REGEX = r"{}(\w+)\([^()]+\)".format(DEFINE_NAME)
FUNCTION_DEFINITION_REGEX = r"{}\w*\([^()]+\)".format(DEFINE_NAME)
INSTANCE_CALL_REGEX = r"{}\([^()]+\)"


class StringFunctionBuilder:
    """
    Base class to extract, reformat, and make some operation over the function as string.
    """

    def __init__(self, function_as_reference: Optional[Callable] = None, function_as_string: Optional[str] = None):
        assert (
            function_as_reference is not None or function_as_string is not None
        ), "Either function reference, or function as string should be provided."

        if function_as_reference is not None:
            self.function_as_string = getsource(function_as_reference)
        else:
            self.function_as_string = function_as_string

        # use black refactor to handle the spaces for the input string function
        self.black_mode = black.FileMode(line_length=1000)

    def build(self) -> None:
        """
        Use the black refactor to remove additional spaces, tab spaces, and reformat the `function_as_string`.
        """
        # reformat the source function using black
        self.function_as_string = black.format_str(self.function_as_string, mode=self.black_mode)

    def get_function_header(self) -> str:
        """
        Extract from the function string its header.
        :return: the header of the function with its name and arguments
        """
        function_header = findall(FUNCTION_DEFINITION_REGEX, self.get_coded_function_as_string())[0]
        return " ".join(function_header.replace("\n", "").split())

    def get_function_name(self) -> str:
        """
        Extract the function header and get the name of the function from it.
        :return: name of the function
        """
        function_header = findall(FUNCTION_NAME_REGEX, self.function_as_string)
        assert len(function_header) == 1, "{} is not a definition of a function."
        return function_header[0]

    def get_coded_function_as_string(self) -> str:
        """
        :return: build the `function_as_string`  and return it.
        """
        return self.function_as_string

    def set_coded_function_as_string(self, function_as_string) -> str:
        """
        set function_as_string to a new one
        """
        self.function_as_string = function_as_string


class StringFunctionArgumentUpdater(StringFunctionBuilder):
    """
    Update the argument in a function header, and pass a new arguments to any instance in the function scope based on
    the names of the instances in this function.
    """

    def __init__(self, function_as_reference: Optional[Callable] = None, function_as_string: Optional[str] = None):
        super(StringFunctionArgumentUpdater, self).__init__(function_as_reference, function_as_string)

        self.argument_names = None
        self.build()

    @staticmethod
    def __inject_string_at_index__(injected_string: str, string: str, index: int) -> str:
        """
        Add new string `injected_string` at the `index` in the original `string`.

        :param injected_string: the injected string
        :param string: original string to be modified
        :param index: index which the new string will be added
        :return: updated string after injected the `injected_string`
        """
        return string[:index] + injected_string + string[index:]

    def update_arguments_in_function_header(self, argument_names: List[str], set_non_default: bool = True) -> None:
        """
        Update the header of the function to accept new arguments which are located in `argument_names`.

        :param set_non_default:
        :param argument_names: list of names for new arguments
        """
        self.argument_names = argument_names
        if set_non_default:
            argument_names = [argument_name + "=None" for argument_name in argument_names]
        function_header_as_string = self.get_function_header()
        # concatenate the arguments to the end of the function header, the index [:-2] = `):`
        new_arguments = ", " + ", ".join(argument_names)
        if "*" in function_header_as_string:
            index = function_header_as_string.index("*") - 2
        else:
            index = function_header_as_string.index(")")
        updated_function_header = self.__inject_string_at_index__(new_arguments, function_header_as_string, index)
        self.function_as_string = sub(FUNCTION_DEFINITION_REGEX, updated_function_header, self.function_as_string)

    def update_arguments_passed_to_instances(self, instances_names: List[str]) -> None:
        """
        Update the `function_as_string` by passing new arguments to the instances that have the names in the list
        `instances_names`.

        :param instances_names: list of names that suppose to be found in the function scope
        """
        assert (
            self.argument_names is not None
        ), "argument names should be provided via `update_arguments_in_function_header`"
        for instances_name in instances_names:
            instance_name_call_regex = INSTANCE_CALL_REGEX.format(instances_name)
            matched_names = findall(instance_name_call_regex, self.function_as_string)
            for matched_name in matched_names:
                # reformat the string to be `instance_name = instance_name` and join between them using `,`.
                new_arguments = ", " + ", ".join([*map(lambda x: x + "=" + x, self.argument_names)])
                # inject the reformatted string to the instances at the end of their call
                updated_instance_call = self.__inject_string_at_index__(new_arguments, matched_name, -1)
                self.function_as_string = sub(instance_name_call_regex, updated_instance_call, self.function_as_string)


class StringFunctionRedefiner(StringFunctionBuilder):
    """
    Add, remove lines of code in the target function inside an object.
    """

    def __init__(self, function_as_reference: Optional[Callable] = None, function_as_string: Optional[str] = None):
        super(StringFunctionRedefiner, self).__init__(function_as_reference, function_as_string)
        self.build()

    def add_line_of_code_in_header_scope(self, line_of_code):
        """
        Add a line of code after the function header

        :param line_of_code: line of code to be added to the function
        """
        source_function_as_lines = self.get_coded_function_as_string().split("\n")
        source_function_as_lines.insert(0, "{}".format(line_of_code))
        self.function_as_string = "\n".join(source_function_as_lines)
        self.build()

    def add_line_of_code_at_begin(self, line_of_code):
        """
        Add a line of code after the function header

        :param line_of_code: line of code to be added to the function
        """
        source_function_as_lines = self.get_coded_function_as_string().split("\n")
        matched_values = [
            idx for idx, value in enumerate(source_function_as_lines) if self.get_function_header() in value
        ]
        index = 0 if len(matched_values) == 0 else matched_values[0] + 1
        source_function_as_lines.insert(index, "    {}".format(line_of_code))
        self.function_as_string = "\n".join(source_function_as_lines)
        self.build()

    def add_line_of_code_at_end(self, line_of_code):
        """
        Add a line of code before the return value

        :param line_of_code: line of code to be added to the function
        """
        source_function_as_lines = self.get_coded_function_as_string().split("\n")
        source_function_as_lines.insert(-2, "    {}".format(line_of_code))
        self.function_as_string = "\n".join(source_function_as_lines)
        self.build()

    def remove_line_of_code(self, line_of_code):
        """
        Remove a line of code with an empty line
        :param line_of_code: line of code to be removed
        """
        self.function_as_string = self.get_coded_function_as_string().replace(line_of_code, "")
        self.build()


class BindStringToObjectFunction(StringFunctionBuilder):
    """
    Compile a string function and bind it inside the instance object.
    """

    def __init__(self, function_as_reference: Optional[Callable] = None, function_as_string: Optional[str] = None):
        super(BindStringToObjectFunction, self).__init__(function_as_reference, function_as_string)
        self.build()

    def bind_function_inside_instance(self, instance_object: object):
        """
        compile the function string and inject it inside the instance object.

        :param instance_object: an instance that will be injected with the new function
        :return: updated instance object that contain the new bind function
        """
        import pyq

        # create the new function in the `local_variable_dict` scope
        local_variable_dict = pyq.__dict__
        exec(self.get_coded_function_as_string(), globals(), local_variable_dict)
        function_name = self.get_function_name()
        # bind the new defined function to the `base_model`
        bound_method = local_variable_dict[function_name].__get__(instance_object, instance_object.__class__)
        setattr(instance_object, function_name, bound_method)
        return instance_object
