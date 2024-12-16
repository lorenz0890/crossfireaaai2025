import os.path
import pickle
from pathlib import Path
from typing import Dict, Optional, Union

from torch import Tensor, jit, load, save
from torch.nn import Module

from pyq.io.base import InputOutput


def torch_deep_copy(torch_object: Module) -> Module:
    return pickle.loads(pickle.dumps(torch_object))


class ModelIO(InputOutput):
    # can be improved by using this comment: https://stackoverflow.com/a/59392276/10725769
    def __init__(self, directory: Union[str, Path]):
        """
        :param directory: the directory to read/write the model into it as string or path
        """
        self.directory = directory


class ModelReader(ModelIO):
    def read(self):
        """
        :return: read pytorch model from the file that provided in the constractor
        """
        return load(self.directory)


class ModelWriter(ModelIO):
    def write(
        self,
        model: Module,
        file_name: str,
        model_inputs: Optional[Union[Dict[str, Tensor], Tensor]] = None,
        save_as: Optional[str] = "graph",
    ):
        """
        save the model as weights and biases in a file located at the `path`
        :param model: a pytorch or parsed model that inherent `torch.nn.Module`
        :param file_name: name of the model to save the model with it in the `directory`
        :param model_inputs: input to be able to trace back the whole model
        :param save_as: string to allow to save the whole model as state_dict, graph, or jit
        :return: True if the write operation otherwise, raise an exception
        """
        if not os.path.exists(self.directory):
            self.create_dir(self.directory)

        assert save_as in [
            "state_dict",
            "graph",
            "jit",
        ], "`save_as` should be on of these only: `state_dict`, `graph`, `jit`"

        if save_as == "jit" and model_inputs is None:
            raise ValueError("`model_input` to be able to save model as graph.")

        if save_as == "jit" and model_inputs is not None:
            jit_model = jit.trace(model, model_inputs)
            jit.save(jit_model, self.directory + "/" + file_name)
        elif save_as == "graph":
            save(model, self.directory + "/" + file_name)
        else:
            save(model.state_dict(), self.directory + "/" + file_name)
        return True
