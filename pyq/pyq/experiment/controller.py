import operator
from copy import deepcopy
from os.path import join
from typing import Iterable, Optional

from pyq.experiment.classes_builder import InstancesBuilder
from pyq.experiment.default_values import (ARGUMENTS_NAMES_NAME, CODE_AS_STRING_NAME, DEFAULT_ACTIVATION_WRAPPER_NAME,
                                           DEFAULT_KFOLD_NAME, DEFAULT_LOSS_COEFFICIENT_NAME, DEFAULT_MODEL_EDITOR_NAME,
                                           DEFAULT_SCHEDULER_NAME, DEFAULT_STATE_DICT_NAME, DEFAULT_TEACHER_LOSSES_NAME,
                                           DEFAULT_TEACHER_MODEL_NAME, DEFAULT_Y_TRANSFORM_NAME, INSTANCES_NAMES_NAME,
                                           UPDATE_NESTED_INSTANCE_NAME, YAML_NAMES)
from pyq.io.model import ModelReader
from pyq.io.yaml_files import YamlReader
from pyq.models.editor import setattr_nested
from pyq.paths import PyqPath
from pyq.training.dispatcher import ExperimentDispatcher


def merge_two_dictionaries(source: dict, destination: dict) -> dict:
    """
    Merge two dictionaries recursively, such that the source dictionary will be overridden by destination dictionary
    source: https://stackoverflow.com/a/20666342/10725769
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge_two_dictionaries(value, node)
        else:
            if key in destination and isinstance(destination[key], (list, tuple)):
                destination[key] = list(destination[key]) + list(value)
            else:
                destination[key] = value
    return destination


class ModelEditorController:
    """
    Class to control the model editor to update the forward functions
    """

    def __init__(self, model_editor_cls, model_editor_arguments):
        """
        :param model_editor_cls: class of the model editor
        :param model_editor_arguments: arguments that hold the lists and configuration for the model editor
        """
        self.model_editor_cls = model_editor_cls

        self.is_inserting_code = CODE_AS_STRING_NAME in model_editor_arguments
        self.update_header = ARGUMENTS_NAMES_NAME in model_editor_arguments
        self.update_instance = INSTANCES_NAMES_NAME in model_editor_arguments
        self.update_nested = UPDATE_NESTED_INSTANCE_NAME in model_editor_arguments

        self.inserted_code_list = model_editor_arguments[CODE_AS_STRING_NAME] if self.is_inserting_code else []
        self.new_arguments_list = model_editor_arguments[ARGUMENTS_NAMES_NAME] if self.update_header else []
        self.instance_names_list = model_editor_arguments[INSTANCES_NAMES_NAME] if self.update_instance else []
        self.update_nested_args = model_editor_arguments[UPDATE_NESTED_INSTANCE_NAME] if self.update_nested else []

    def apply(self, model):
        # construct the classes for the model editors
        model_editor = self.model_editor_cls(model)
        if self.update_nested:
            nested_instances_editors = [
                self.model_editor_cls(operator.attrgetter(instance_name)(model))
                for instance_name in self.update_nested_args
            ]

        # insert code in the forward function scope (above the definition of the function)
        if self.is_inserting_code:
            model_editor.insert_code_in_forward_scope(self.inserted_code_list)
            if self.update_nested:
                for nested_instances_editor in nested_instances_editors:
                    nested_instances_editor.insert_code_in_forward_scope(self.inserted_code_list)

        # inject new arguments in the function signature, and pass them to some instances in the forward implementation
        if self.update_header and self.update_instance:
            model_editor.update_arguments_for_forward_function(self.new_arguments_list, self.instance_names_list)
            if self.update_nested:
                for nested_instances_editor in nested_instances_editors:
                    nested_instances_editor.update_arguments_for_forward_function(
                        self.new_arguments_list, self.instance_names_list
                    )

        # insert code in the forward function implementation (inside the definition of the function)
        if self.is_inserting_code:
            model_editor.insert_code_in_forward(self.inserted_code_list)
            if self.update_nested:
                for nested_instances_editor in nested_instances_editors:
                    nested_instances_editor.insert_code_in_forward(self.inserted_code_list)

        # set the updated forward function inside the model, and its nested instances updates
        model = model_editor.get_torch_model()
        if self.update_nested:
            for path, nested_instances_editor in zip(*[self.update_nested_args, nested_instances_editors]):
                setattr_nested(model, path, nested_instances_editor.get_torch_model())
        return model


class ExperimentController:
    """
    Interface or API to handle the experiments using the dispatcher by controlling setup, running, and saving of the
    experiment. The controller only accepts the device, yaml file, and checkpoint path.
    """

    def __init__(self, device, logging_dir: str = None):
        """
        :param device: torch device to detect from it if the gpu should be used or not
        """
        self.experiment_dispatcher = ExperimentDispatcher(logging_dir)
        self.builder = InstancesBuilder()
        self.device = device

    def __set_experiment_attribute__(self, classes_as_dict):
        """
        set the attributes inside the class with name equal to dictionary key, and its value equal to a dictionary value
        :param classes_as_dict: target dictionary with values holding objects
        """

        for key in YAML_NAMES:
            if key in classes_as_dict:
                self.__setattr__(key, classes_as_dict[key])

        # handle the case when the wrapper has its arguments as dict
        self.layer_wrapper, self.layer_wrapper_arguments = (
            self.layer_wrapper if isinstance(self.layer_wrapper, tuple) else (self.layer_wrapper, {})
        )

        self.activation_wrapper, self.activation_wrapper_arguments = (
            (self.activation_wrapper if isinstance(self.activation_wrapper, tuple) else (self.activation_wrapper, {}))
            if hasattr(self, DEFAULT_ACTIVATION_WRAPPER_NAME)
            else (None, None)
        )

        # handle the pre-trained loaded model
        self.model = self.model.read() if isinstance(self.model, ModelReader) else self.model

        # handle the model editor case
        if hasattr(self, DEFAULT_MODEL_EDITOR_NAME):
            model_editor_class, model_editor_args = self.model_editor
            model_editor_controller = ModelEditorController(model_editor_class, model_editor_args)
            self.model = model_editor_controller.apply(self.model)

        # handle the case of teacher model
        self.teacher_model = self.teacher_model if hasattr(self, DEFAULT_TEACHER_MODEL_NAME) else None
        self.teacher_model = (
            self.teacher_model.read() if isinstance(self.teacher_model, ModelReader) else self.teacher_model
        )
        self.teacher_losses = self.teacher_losses if hasattr(self, DEFAULT_TEACHER_LOSSES_NAME) else None
        self.loss_coefficient = self.loss_coefficient if hasattr(self, DEFAULT_LOSS_COEFFICIENT_NAME) else None

        # handle the case of y data type
        self.y_transform = self.y_transform if hasattr(self, DEFAULT_Y_TRANSFORM_NAME) else None

        # handle the case of kfold spliter
        self.kfold_spliter = self.kfold if hasattr(self, DEFAULT_KFOLD_NAME) else None

        # handle the case when the optimizer has its arguments as dict
        self.optimizer, self.optimizer_args = (
            self.optimizer if isinstance(self.optimizer, Iterable) else (self.optimizer, {})
        )
        self.optimizer_state_dict = None
        if DEFAULT_STATE_DICT_NAME in self.optimizer_args:
            if isinstance(self.optimizer_args[DEFAULT_STATE_DICT_NAME], ModelReader):
                self.optimizer_state_dict = self.optimizer_args.pop(DEFAULT_STATE_DICT_NAME).read()

        # handle the case when the scheduler has its arguments as dict
        self.scheduler = self.scheduler if hasattr(self, DEFAULT_SCHEDULER_NAME) else None
        self.scheduler, self.scheduler_args = (
            self.scheduler if isinstance(self.scheduler, Iterable) else (self.scheduler, {})
        )
        self.scheduler_state_dict = None
        if DEFAULT_STATE_DICT_NAME in self.scheduler_args:
            if isinstance(self.scheduler_args[DEFAULT_STATE_DICT_NAME], ModelReader):
                self.scheduler_state_dict = self.scheduler_args.pop(DEFAULT_STATE_DICT_NAME).read()

        # handle the case of the training loop arguments
        self.training_loop, self.training_loop_args = (
            self.training_loop if isinstance(self.training_loop, Iterable) else (self.training_loop, {})
        )

    def setup_experiment_parameters(self, yaml_file_path: str):
        """
        setup main attribute and then initialize the experiment dispatcher
        :param yaml_file_path: the main file to set up the experiment
        """
        default_yaml_file_path = join(PyqPath.get_pyq_package_dir(), "default_configuration_values.yaml")
        default_classes_as_dict_str = YamlReader(default_yaml_file_path).read()
        self.read_classes_as_dict_str = YamlReader(yaml_file_path).read()
        classes_as_dict_str = merge_two_dictionaries(self.read_classes_as_dict_str, default_classes_as_dict_str)

        classes_as_dict_obj = self.builder.convert_dictionary_to_instances(deepcopy(classes_as_dict_str))

        self.__set_experiment_attribute__(classes_as_dict_obj)
        self.experiment_dispatcher.initialize(
            dataset_initializer=self.dataset,
            dataloader_args=self.dataloader,
            kfold_spliter=self.kfold_spliter,
            model=self.model,
            teacher_model=self.teacher_model,
            teacher_losses=self.teacher_losses,
            loss_coefficient=self.loss_coefficient,
            task=self.task,
            parser=self.parser,
            layer_wrapper=self.layer_wrapper,
            layer_wrapper_arguments=self.layer_wrapper_arguments,
            activation_wrapper=self.activation_wrapper,
            activation_wrapper_arguments=self.activation_wrapper_arguments,
            losses=self.losses,
            y_transform=self.y_transform,
            metrics=self.metrics,
            optimizer=self.optimizer,
            optimizer_arguments=self.optimizer_args,
            optimizer_state_dict=self.optimizer_state_dict,
            scheduler=self.scheduler,
            scheduler_arguments=self.scheduler_args,
            scheduler_state_dict=self.scheduler_state_dict,
            training_loop=self.training_loop,
        )

    def dispatch_experiment(self, ckpt_path: Optional[str] = None):
        """
        start or resume the experiment using dispatcher by training and evaluate the pytorch lightning trainer
        :param ckpt_path: path for the trainer checkpoint in case the experiment will resume
        """
        self.experiment_dispatcher.train_and_evaluate(
            device=self.device, ckpt_directory=ckpt_path, **self.training_loop_args
        )

    def finalize_experiment(self):
        """
        save the model and pytorch lightning trainer checkpoint
        """
        self.experiment_dispatcher.save(yaml_as_dict=self.read_classes_as_dict_str)
