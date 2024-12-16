import copy
from unittest import TestCase, main

import torch

from pyq.experiment.classes_builder import InstancesBuilder
from pyq.io.yaml_files import YamlReader
from pyq.paths import PyqPath
from pyq.training.dispatcher import ExperimentDispatcher
from pyq.utils import equal_state_dictionary


class TestMnistExperimentDispatcher(TestCase):
    def setUp(self):
        self.experiment_dispatcher = ExperimentDispatcher()
        self.builder = InstancesBuilder()

        yaml_file_path = "./test_cases/dispatcher_mnist_test_case.yaml"
        class_as_dict = YamlReader(yaml_file_path).read()
        objects_as_dict = self.builder.convert_dictionary_to_instances(class_as_dict)

        self.dataset_initializer = objects_as_dict["dataset"]
        self.dataloader = objects_as_dict["dataloader"]
        self.model = objects_as_dict["model"]
        self.task = objects_as_dict["task"]
        self.parser = objects_as_dict["parser"]
        self.layer_wrapper = objects_as_dict["layer_wrapper"]
        self.losses = objects_as_dict["losses"]
        self.optimizer, self.optimizer_args = objects_as_dict["optimizer"]
        self.scheduler, self.scheduler_args = objects_as_dict["scheduler"]
        self.training_loop, self.training_loop_args = objects_as_dict["training_loop"]

        self.experiment_dispatcher = ExperimentDispatcher()

    def test_experiment_dispatcher_initialize(self):
        self.experiment_dispatcher.initialize(
            dataset_initializer=self.dataset_initializer,
            kfold_spliter=None,
            dataloader_args=self.dataloader,
            model=self.model,
            teacher_model=None,
            teacher_losses=None,
            loss_coefficient=None,
            task=self.task,
            parser=self.parser,
            layer_wrapper=self.layer_wrapper,
            layer_wrapper_arguments={},
            activation_wrapper=self.layer_wrapper,
            activation_wrapper_arguments={},
            losses=self.losses,
            y_transform={},
            metrics=[],
            optimizer=self.optimizer,
            optimizer_arguments=self.optimizer_args,
            optimizer_state_dict=None,
            scheduler=self.scheduler,
            scheduler_arguments=self.scheduler_args,
            scheduler_state_dict=None,
            training_loop=self.training_loop,
        )

    def test_experiment_dispatcher_train_and_evaluate(self):
        self.test_experiment_dispatcher_initialize()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.experiment_dispatcher.train_and_evaluate(device=device, **self.training_loop_args)
        self.experiment_dispatcher.save()
        self.experiment_dispatcher.clear_logs()

        pyq_model = self.experiment_dispatcher.model
        torch_model = self.experiment_dispatcher.get_unparsed_model(pyq_model)
        self.assertTrue(equal_state_dictionary(pyq_model.state_dict(), torch_model.state_dict()))


class TestCoraExperimentDispatcher(TestCase):
    def setUp(self):
        self.experiment_dispatcher = ExperimentDispatcher()
        self.builder = InstancesBuilder()

        yaml_file_path = "./test_cases/dispatcher_cora_test_case.yaml"
        class_as_dict = YamlReader(yaml_file_path).read()
        objects_as_dict = self.builder.convert_dictionary_to_instances(class_as_dict)

        self.dataset_initializer = objects_as_dict["dataset"]
        self.dataloader = objects_as_dict["dataloader"]
        self.model = objects_as_dict["model"]
        self.task = objects_as_dict["task"]
        self.parser = objects_as_dict["parser"]
        self.layer_wrapper = objects_as_dict["layer_wrapper"]
        self.losses = objects_as_dict["losses"]
        self.optimizer, self.optimizer_args = objects_as_dict["optimizer"]
        self.scheduler, self.scheduler_args = objects_as_dict["scheduler"]
        self.training_loop, self.training_loop_args = objects_as_dict["training_loop"]

        self.experiment_dispatcher = ExperimentDispatcher()

    def test_experiment_dispatcher_initialize(self):
        self.experiment_dispatcher.initialize(
            dataset_initializer=self.dataset_initializer,
            kfold_spliter=None,
            dataloader_args=self.dataloader,
            model=self.model,
            teacher_model=None,
            teacher_losses=None,
            loss_coefficient=None,
            task=self.task,
            parser=self.parser,
            layer_wrapper=self.layer_wrapper,
            layer_wrapper_arguments={},
            activation_wrapper=self.layer_wrapper,
            activation_wrapper_arguments={},
            losses=self.losses,
            y_transform={},
            metrics=[],
            optimizer=self.optimizer,
            optimizer_arguments=self.optimizer_args,
            optimizer_state_dict=None,
            scheduler=self.scheduler,
            scheduler_arguments=self.scheduler_args,
            scheduler_state_dict=None,
            training_loop=self.training_loop,
        )

    def test_experiment_dispatcher_train_and_evaluate(self):
        self.test_experiment_dispatcher_initialize()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.experiment_dispatcher.train_and_evaluate(device=device, **self.training_loop_args)
        self.experiment_dispatcher.save()
        self.experiment_dispatcher.clear_logs()

        pyq_model = self.experiment_dispatcher.model
        torch_model = self.experiment_dispatcher.get_unparsed_model(pyq_model)
        self.assertTrue(equal_state_dictionary(pyq_model.state_dict(), torch_model.state_dict()))


if __name__ == "__main__":
    main()
