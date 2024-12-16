import os.path
from pathlib import Path
from typing import Union

import yaml

from pyq.io.base import InputOutput


class YamlIO(InputOutput):
    def __init__(self, directory: Union[str, Path]):
        """
        :param directory: the directory and the yaml file name concatenated together as string or path
        """
        self.directory = directory


class YamlReader(YamlIO):
    """
    Naive class to read the yaml file and convert it to dictionary.
    """

    def read(self) -> dict:
        """
        :return: dictionary that holds the same structure of the yaml file
        """
        with open(self.directory, "r") as yaml_file:
            return yaml.safe_load(yaml_file)


class YamlWriter(YamlIO):
    """
    Basic yaml file handler to write new data from dictionary to a yaml file.
    """

    def write(self, yaml_as_dict: dict, file_name: str) -> bool:
        """
        print out the dictionary in a file located at the `path`
        :param yaml_as_dict: a dictionary that holds strings and lists only!
        :param file_name: name of the yaml file that will be written in the `directory`
        :return: True if the write operation otherwise, raise an exception
        """
        if not os.path.exists(self.directory):
            self.create_dir(self.directory)
        with open(os.path.join(self.directory, file_name), "w") as yaml_file:
            yaml.dump(yaml_as_dict, yaml_file, sort_keys=False, default_flow_style=False)
            return True

    def append(self, new_yaml_as_dict: dict, file_name: str) -> dict:
        """
        concatenate new dictionary to a yaml file that already exists
        :param new_yaml_as_dict: dictionary to update the yaml file with its parameters
        :param file_name: name of the yaml file that will append new data in it
        :return: the updated version of the yaml as a dictionary
        """
        yaml_as_dict = YamlReader(self.directory).read()
        new_yaml_as_dict.update(yaml_as_dict)
        self.write(yaml_as_dict=new_yaml_as_dict, file_name=file_name)
        return new_yaml_as_dict
