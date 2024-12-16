import importlib
from collections.abc import Iterable
from typing import Dict, Union

from pyq.experiment.default_values import DEFAULT_CLASS_NAME, LISTED_CLASS_NAME_TYPE, UNBUILD_CLASS_FLAG


def evaluate_dictionary(dict_hold_strings: dict):
    """
    execute the values in the dictionary to obtain the correct datatype instead of string

    :param dict_hold_strings: dictionary that has in its value parameters
    """
    for key in dict_hold_strings:
        try:
            dict_hold_strings[key] = eval(dict_hold_strings[key])
        except NameError:
            pass
        except SyntaxError:
            pass
        except TypeError:
            pass
    return dict_hold_strings


class InstancesBuilder:
    """
    Builder for the classes instances from a dictionary, by iterating over the values of the dictionary.
    """

    def __init__(self, import_from_dir: Union[None, str] = "pyq"):
        """
        import all the modules from the provided path `import_from_dir` to use it to build the instances

        :param import_from_dir: path to import classes from
        """
        import_from_dir = import_from_dir.replace("/", ".")
        self.module = importlib.import_module(import_from_dir)

    def construct_instance(self, unbuild_instance: LISTED_CLASS_NAME_TYPE):
        """
        recursively iterate over a dictionary to construct objects of its string names, and then return the main
        instance that accepted in its constructor the objects/list of objects that were built recursively

        :param unbuild_instance: a dictionary that has in its values string (name of a class), list of string,
                                 or another dictionary that follows the same rules
        :return: a constructed class that was built based on the strings in the dictionary

        """
        # if the dictionary has no `class_name` return it as dictionary
        if isinstance(unbuild_instance, Iterable) and DEFAULT_CLASS_NAME not in unbuild_instance:
            return unbuild_instance

        # if the unbuild instance is not dictionary return it
        if not isinstance(unbuild_instance, dict):
            return unbuild_instance

        # pop out from the dictionary the class name and store it to build it
        class_name = unbuild_instance.pop(DEFAULT_CLASS_NAME)
        # evaluate the parameter in the dictionary values to convert string to int, tuple, list, ... etc
        unbuild_instance = evaluate_dictionary(unbuild_instance)

        if isinstance(unbuild_instance, dict):
            for key in unbuild_instance:

                # check nested classes to build
                if isinstance(unbuild_instance[key], dict) and DEFAULT_CLASS_NAME in unbuild_instance[key]:
                    unbuild_instance[key] = self.construct_instance(unbuild_instance[key])

                # check list of objects to build
                if isinstance(unbuild_instance[key], list):
                    unbuild_instance[key] = [
                        self.construct_instance(item_to_build) for item_to_build in unbuild_instance[key]
                    ]

        if UNBUILD_CLASS_FLAG in class_name:
            # remove `$` from class name and return the class without constructing it as `type` only
            class_name = class_name.replace(UNBUILD_CLASS_FLAG, "")
            class_to_construct = getattr(self.module, class_name)
            # return the additional arguments as dictionary if it's exist
            if len(unbuild_instance) != 0:
                return class_to_construct, unbuild_instance
            return class_to_construct
        else:
            # construct the class and return it as an object
            class_to_construct = getattr(self.module, class_name)
            return class_to_construct(**unbuild_instance)

    def convert_dictionary_to_instances(self, dictionary_as_strings: Dict[str, LISTED_CLASS_NAME_TYPE]) -> dict:
        """
        iterate over the values in the provided dictionary to build instance for each one of them

        :param dictionary_as_strings: dictionary that has in its value another dictionaries to build instances from them
        :return: dictionary that has in its values object or instances that were built dynamically
        """
        for key in dictionary_as_strings:
            # iterate over list to build objects for each item in the list
            if type(dictionary_as_strings[key]) is list:
                dictionary_as_strings[key] = [self.construct_instance(key_i) for key_i in dictionary_as_strings[key]]
            elif type(dictionary_as_strings[key]) is dict:
                if DEFAULT_CLASS_NAME in dictionary_as_strings[key]:
                    dictionary_as_strings[key] = self.construct_instance(dictionary_as_strings[key])
                else:
                    dictionary_as_strings[key] = {
                        key_i: self.construct_instance(dictionary_as_strings[key][key_i])
                        for key_i in dictionary_as_strings[key]
                    }
            else:
                raise ValueError(
                    "unsupported type {} in the dictionary with the value {}".format(
                        type(dictionary_as_strings[key]), dictionary_as_strings[key]
                    )
                )
        return dictionary_as_strings
