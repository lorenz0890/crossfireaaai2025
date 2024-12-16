from os import listdir
from os.path import join
from pathlib import Path
from typing import Union

from pytorch_lightning import Trainer
from torch import device
from torch.nn import Module

from pyq.experiment.default_values import (PYLIGHTNING_CHECKPOINT_EXTENSION, PYLIGHTNING_CHECKPOINT_FILE_NAME,
                                           PYQ_LOGS_DIR_NAME, PYQ_MODEL_FILE_NAME, PYQ_OPTIMIZER_FILE_NAME,
                                           PYQ_SCHEDULER_FILE_NAME, PYQ_TORCH_MODEL_EXTENSION, PYQ_YAML_FILE_NAME,
                                           TORCH_OPTIMIZAR_EXTENSION, TORCH_SCHEDULER_EXTENSION, YAML_EXTENSION)
from pyq.io.base import InputOutput
from pyq.io.model import ModelWriter
from pyq.io.yaml_files import YamlWriter

CPU = "cpu"


class ExperimentIO(InputOutput):
    def __init__(self, directory: Union[str, Path]):
        """
        :param directory: the directory as string or path to save experiment logs inside it
        """
        self.directory = directory


class ExperimentReader(ExperimentIO):
    def _find_file_with_extension(self, extension: str):
        files_in_directory = listdir(self.directory)
        if len(files_in_directory) == 0:
            raise ValueError("there is no files in the provided directory {}".format(self.directory))

        files_in_directory_with_extension = list(filter(lambda k: extension in k, files_in_directory))
        if len(files_in_directory_with_extension) == 0:
            raise ValueError(
                "there is no files with extension {} in the provided directory {}".format(extension, self.directory)
            )

        if len(files_in_directory_with_extension) > 1:
            raise ValueError(
                "there are a lot of files with extension {} in the provided directory {}".format(
                    extension, self.directory
                )
            )

        return join(self.directory, files_in_directory_with_extension[0])

    def find_model_path(self):
        return self._find_file_with_extension(PYQ_TORCH_MODEL_EXTENSION)

    def find_optimizer_path(self):
        return self._find_file_with_extension(TORCH_OPTIMIZAR_EXTENSION)

    def find_scheduler_path(self):
        return self._find_file_with_extension(TORCH_SCHEDULER_EXTENSION)

    def find_checkpoint_path(self):
        return self._find_file_with_extension(PYLIGHTNING_CHECKPOINT_EXTENSION)

    def read(self):
        # override the `restore_model` function and restore model manually
        # trainer.checkpoint_connector.restore_model = lambda: None

        checkpoint_path = self.find_checkpoint_path()
        model_path = self.find_model_path()
        optimizer_path = self.find_optimizer_path()
        scheduler_path = self.find_scheduler_path()

        return checkpoint_path, model_path, optimizer_path, scheduler_path


class ExperimentWriter(ExperimentIO):
    def write(
        self,
        trainer: Trainer,
        pyq_model: Module,
        optimizer: Module,
        scheduler: Module,
        yaml_as_dict: dict,
        unique_string: str,
    ):
        pyq_dir = join(self.directory, PYQ_LOGS_DIR_NAME, unique_string)
        self.create_dir(pyq_dir)

        check_point_file_path = PYLIGHTNING_CHECKPOINT_FILE_NAME + PYLIGHTNING_CHECKPOINT_EXTENSION
        model_file_path = PYQ_MODEL_FILE_NAME + PYQ_TORCH_MODEL_EXTENSION
        optimizer_file_path = PYQ_OPTIMIZER_FILE_NAME + TORCH_OPTIMIZAR_EXTENSION
        scheduler_file_path = PYQ_SCHEDULER_FILE_NAME + TORCH_SCHEDULER_EXTENSION
        yaml_file_path = PYQ_YAML_FILE_NAME + YAML_EXTENSION

        trainer.save_checkpoint(join(pyq_dir, check_point_file_path))
        ModelWriter(pyq_dir).write(pyq_model.to(device(CPU)), model_file_path, save_as="graph")
        ModelWriter(pyq_dir).write(optimizer, optimizer_file_path)
        ModelWriter(pyq_dir).write(scheduler, scheduler_file_path)
        YamlWriter(pyq_dir).write(yaml_as_dict, yaml_file_path)
        return pyq_dir
