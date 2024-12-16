import shutil
from copy import deepcopy
from datetime import datetime
from os import getcwd
from os.path import exists, join
from typing import List, Optional, OrderedDict, Union

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler
from torch._C import device as _device
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import SubsetRandomSampler
from torchmetrics.metric import Metric

from pyq.core.parser import TorchModelParser
from pyq.core.wrapper import TorchModuleWrapper
from pyq.datasets.initializer import DataInitializer
from pyq.datasets.splitter import DataKFold
from pyq.experiment.default_values import (DEFAULT_KFOLD_NAME, PYLIGHTNING_LOGS_DIR_NAME, PYQ_EXPERIMENTS_DIRECTORY,
                                           TORCH_MODEL_EXTENSION, TORCH_MODEL_FILE_NAME)
from pyq.io.base import InputOutput
from pyq.io.experiment import CPU, ExperimentReader, ExperimentWriter
from pyq.io.model import ModelWriter
from pyq.training.loop import TrainingLoop
from pyq.training.task import Task


class ExperimentDispatcher:
    """
    Dispatching the experiment by `initialize`, `train_and_evaluate`, and then `save` the mode and the yaml file that
    passed to run this experiment to run it, this dispatcher supposed to use by the experiment controller.
    """

    def __init__(self, logging_dir: str = None):
        """
        :param logging_dir: path to export loggers, by default it will be beside the project directory in `experiments`
        """
        if not logging_dir:
            self.logging_dir = join(getcwd(), PYQ_EXPERIMENTS_DIRECTORY)
            InputOutput.create_dir(self.logging_dir)
        else:
            self.logging_dir = logging_dir

        self.date_time_string = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S.%f"))

    def initialize(
        self,
        dataset_initializer: DataInitializer,
        kfold_spliter: Union[DataKFold, None],
        dataloader_args: dict,
        model: Module,
        teacher_model: Union[Module, None],
        teacher_losses: List[_Loss],
        loss_coefficient: float,
        task: Task,
        parser: TorchModelParser,
        layer_wrapper: TorchModuleWrapper,
        layer_wrapper_arguments: dict,
        activation_wrapper: TorchModuleWrapper,
        activation_wrapper_arguments: dict,
        losses: List[_Loss],
        y_transform: Union[dict, dict],
        metrics: List[Metric],
        optimizer: Union[Optimizer, type],
        optimizer_arguments: dict,
        optimizer_state_dict: Union[OrderedDict, None],
        scheduler: Union[Optimizer, type],
        scheduler_arguments: dict,
        scheduler_state_dict: Union[OrderedDict, None],
        training_loop: Union[TrainingLoop, type],
    ):
        """
        set the essential components for the experiment, and store them inside the `ExperimentDispatcher` class

        :param dataset_initializer: see pyq.datasets.initializer documentation
        :param kfold_spliter:
        :param dataloader_args: the dataloader arguments to create dataloader based on them
        :param model: target model to train, it could be native pytorch model or pytorch geometric model
        :param teacher_model:  pre-trained huge model
        :param loss_coefficient:
        :param teacher_losses:
        :param task: see pyq.training.task documentation
        :param parser: see pyq.core.parser documentation
        :param layer_wrapper: see pyq.core.wrapper documentation
        :param layer_wrapper_arguments: the wrapper arguments
        :param activation_wrapper: see pyq.core.wrapper documentation
        :param activation_wrapper_arguments: the wrapper arguments
        :param losses: list of the loss functions that will be summed up to optimize of it
        :param y_transform: data type to cast the y to it
        :param metrics: list of metric to evaluate and plot them on the tensorboard
        :param optimizer: the optimizer that tunes the learnable parameters in the `model`
        :param optimizer_arguments: the optimizer arguments to create new optimizer based on them
        :param optimizer_state_dict: the state for a saved optimizer
        :param scheduler: the scheduler for the optimizer that tunes the hyper-parameters for the `optimizer`
        :param scheduler_arguments: the scheduler arguments to create new scheduler based on them
        :param scheduler_state_dict: the state for a saved scheduler
        :param training_loop: see pyq.training.loop documentation
        """
        # basic initialization for running experiment
        self.losses = losses
        self.teacher_losses = teacher_losses
        self.loss_coefficient = loss_coefficient
        self.y_transform = y_transform
        self.metrics = metrics
        self.training_loop = training_loop
        self.kfold_spliter = kfold_spliter
        self.dataloader_args = dataloader_args

        # empty list to save the kfolds state dictionaries
        self.kfolds_trainer_state_dicts = []
        self.kfolds_models_state_dicts = []
        self.kfolds_optimizers_state_dicts = []
        self.kfolds_schedulers_state_dicts = []

        # initialize the training and validation datasets
        self.training_dataset, self.validation_dataset = dataset_initializer.get_train_test_set()
        self.is_kfold_splitted = False if kfold_spliter is None else True
        self.kfold_spliter.set_dataset(self.training_dataset) if self.is_kfold_splitted else None

        # create unique experiment name
        dataset_name = dataset_initializer.full_dataset_name
        model_name = model.__class__.__name__.lower()
        self.experiment_name = "_".join([model_name, dataset_name, self.date_time_string])

        # construct the parser
        skip_parsing = parser.skip_parsing
        skip_layer = parser.skip_layer_by_type
        skip_names = parser.skip_layer_by_regex
        delete_layer = parser.delete_layer_by_type
        remove_layers_bias = parser.remove_layers_bias
        self.parser = parser.__class__(
            callable_object=layer_wrapper,
            callable_object_kwargs=layer_wrapper_arguments,
            callable_object_for_nonparametric=activation_wrapper,
            callable_object_for_nonparametric_kwargs=activation_wrapper_arguments,
            remove_layers_bias=remove_layers_bias,
            skip_layer_by_type=skip_layer,
            skip_layer_by_regex=skip_names,
            delete_layer_by_type=delete_layer,
            skip_parsing=skip_parsing,
        )

        # construct the unparser
        self.unparser = parser.__class__(
            callable_object=layer_wrapper.unwrap,
            callable_object_for_nonparametric=activation_wrapper.unwrap if activation_wrapper else None,
            remove_layers_bias=remove_layers_bias,
            skip_layer_by_type=skip_layer,
            skip_layer_by_regex=skip_names,
            skip_parsing=skip_parsing,
        )

        # parse the model
        self.model = self.parser.apply(model)
        self.teacher_model = teacher_model

        # task definition
        self.task = task.__class__(task_name=task.task_name, model=model, dataset=self.training_dataset)

        # check that the task, dataset, and model are computable with each other
        if not self.task.is_dataset_compatible_with_task():
            raise ValueError(
                "{} dataset is not compatible with the task {}".format(
                    self.training_dataset.__class__.__name__, task.__class__.__name__
                )
            )

        if not self.task.is_dataset_compatible_with_model():
            raise ValueError(
                "{} dataset is not compatible with the model {}".format(
                    self.training_dataset.__class__.__name__, model.__class__.__name__
                )
            )

        # assign the optimizer argument, optimizer, and load its status if it's available
        self.optimizer_arguments = optimizer_arguments
        self.optimizer = optimizer(self.model.parameters(), **optimizer_arguments)
        self.optimizer.load_state_dict(optimizer_state_dict) if optimizer_state_dict else None

        # assign the scheduler argument, scheduler, and load its status if it's available
        self.scheduler_arguments = scheduler_arguments
        self.scheduler = scheduler(self.optimizer, **scheduler_arguments)
        self.scheduler.load_state_dict(scheduler_state_dict) if scheduler_state_dict else None

        # set single input for the model
        # TODO (Samir): remove all add arguments by the model editor
        self.model_inputs = self.task.get_model_inputs()

    def train_and_evaluate(
        self,
        epoch: int,
        device: _device,
        accelerator: Optional[str] = None,
        ckpt_directory: Optional[str] = None,
        use_profiler: Optional[bool] = False,
        deterministic: Optional[bool] = False,
        precision: Optional[Union[int, str]] = 32,
        callbacks: Optional[List[Callback]] = None,
        enable_checkpointing: Optional[bool] = False,
        number_of_gpus: Optional[Union[List[int], str, int]] = -1,
    ):
        """
        start training and evaluation of the model for the specific number of epoch

        :param epoch: number of epochs to train and evaluate the model on dataset
        :param device: initialized device from pytorch to run experiment on gpus
        :param precision: double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16).
        :param deterministic: sets whether PyTorch operations must use deterministic algorithms
        :param callbacks: add a callback or list of callbacks
        :param enable_checkpointing: it will configure a default ModelCheckpoint callback
        :param ckpt_directory: optional parameter to allow resuming experiment
        :param accelerator: optional parameter to determine data parallel technique
        :param number_of_gpus: number of GPUs to train on (int) or which GPUs to train on (list or str)
        :param use_profiler: if True then there will be profiler logs in the pyq logs directory
        """
        ckpt_path = None
        gpus = number_of_gpus if str(device).startswith("cuda") else 0

        if (
            not self.model
            or not self.task
            or (not self.losses and not self.teacher_losses)
            or not self.optimizer
            or not self.scheduler
            or not self.training_loop
        ):
            raise ValueError("call the function `initialize` from the class {}".format(self.__class__.__name__))

        if use_profiler and gpus != 1:
            raise ValueError(
                "can't collect the profiler logs during running parallel computation on the GPUs, "
                "set `number_of_gpus` to be 1"
            )

        if ckpt_directory:
            ckpt_path, _, _, _ = ExperimentReader(ckpt_directory).read()
            print("Model loaded correctly from {}.".format(ckpt_directory))

        # if there's hanging happen somehow during the training, change the GPUs from `-1` to `1` to fix it

        if self.is_kfold_splitted:
            training_indices, validation_indices = self.kfold_spliter.get_indices_list()

            # save the default state dictionary to reset the status for each fold
            default_model_state_dict = deepcopy(self.model.state_dict())
            default_optimizer_state_dict = deepcopy(self.optimizer.state_dict())
            default_scheduler_state_dict = deepcopy(self.scheduler.state_dict())

            models_list = [deepcopy(self.model) for _ in range(len(training_indices))]
            optimizers_list = [
                self.optimizer.__class__(model_i.parameters(), **self.optimizer_arguments) for model_i in models_list
            ]
            schedulers_list = [
                self.scheduler.__class__(optimizers_i, **self.scheduler_arguments) for optimizers_i in optimizers_list
            ]

            for i, (training_index, validation_index, model, optimizer, scheduler) in enumerate(
                zip(training_indices, validation_indices, models_list, optimizers_list, schedulers_list)
            ):
                experiment_kfold_name = join(self.experiment_name, "{}_{}".format(DEFAULT_KFOLD_NAME, str(i)))

                tensorboard_logger = self.get_tensorboard_logger(experiment_kfold_name)
                profiler = self.get_profiler(experiment_kfold_name) if use_profiler else None

                # reset the status of the mode, optimizer, and scheduler
                model.load_state_dict(default_model_state_dict)
                optimizer.load_state_dict(default_optimizer_state_dict)
                scheduler.load_state_dict(default_scheduler_state_dict)

                # build the sampler form the k-fold index
                training_sampler = SubsetRandomSampler(training_index)
                validation_sampler = SubsetRandomSampler(validation_index)

                training_loader = self.task.get_dataloader_class()(
                    self.training_dataset, **self.dataloader_args, sampler=training_sampler
                )
                validation_loader = self.task.get_dataloader_class()(
                    self.training_dataset, **self.dataloader_args, sampler=validation_sampler
                )

                training_loop_lightning_arguments = {
                    "task": self.task,
                    "model": model,
                    "losses": self.losses,
                    "y_transform": self.y_transform,
                    "metrics": self.metrics,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                }
                if self.teacher_model:
                    training_loop_lightning_arguments.update({"teacher_model": self.teacher_model})
                    training_loop_lightning_arguments.update({"teacher_losses": self.teacher_losses})
                    training_loop_lightning_arguments.update({"loss_coefficient": self.loss_coefficient})

                self.__fit__(
                    # trainer arguments
                    gpus=gpus,
                    epoch=epoch,
                    precision=precision,
                    profiler=profiler,
                    callbacks=callbacks,
                    accelerator=accelerator,
                    tensorboard_logger=tensorboard_logger,
                    deterministic=deterministic,
                    logging_dir=self.logging_dir,
                    enable_checkpointing=enable_checkpointing,
                    # fitting arguments
                    ckpt_path=ckpt_path,
                    training_loader=training_loader,
                    validation_loader=validation_loader,
                    # training loop lightning arguments
                    training_loop_lightning_arguments=training_loop_lightning_arguments,
                )

                # store the current state dict to save the experiment model and state
                self.kfolds_trainer_state_dicts.append(self.trainer.__dict__.copy())
                self.kfolds_models_state_dicts.append(deepcopy(model.state_dict()))
                self.kfolds_optimizers_state_dicts.append(deepcopy(optimizer.state_dict()))
                self.kfolds_schedulers_state_dicts.append(deepcopy(scheduler.state_dict()))

        else:
            tensorboard_logger = self.get_tensorboard_logger(self.experiment_name)
            profiler = self.get_profiler(self.experiment_name) if use_profiler else None

            training_loader = self.task.get_dataloader_class()(self.training_dataset, **self.dataloader_args)
            validation_loader = self.task.get_dataloader_class()(self.validation_dataset, **self.dataloader_args)

            training_loop_lightning_arguments = {
                "task": self.task,
                "model": self.model,
                "losses": self.losses,
                "y_transform": self.y_transform,
                "metrics": self.metrics,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
            }
            if self.teacher_model:
                training_loop_lightning_arguments.update({"teacher_model": self.teacher_model})
                training_loop_lightning_arguments.update({"teacher_losses": self.teacher_losses})
                training_loop_lightning_arguments.update({"loss_coefficient": self.loss_coefficient})

            self.__fit__(
                # trainer arguments
                gpus=gpus,
                epoch=epoch,
                precision=precision,
                profiler=profiler,
                callbacks=callbacks,
                accelerator=accelerator,
                tensorboard_logger=tensorboard_logger,
                deterministic=deterministic,
                logging_dir=self.logging_dir,
                enable_checkpointing=enable_checkpointing,
                # fitting arguments
                ckpt_path=ckpt_path,
                training_loader=training_loader,
                validation_loader=validation_loader,
                # training loop lightning arguments
                training_loop_lightning_arguments=training_loop_lightning_arguments,
            )

    def __fit__(
        self,
        # trainer arguments
        gpus,
        epoch,
        precision,
        profiler,
        callbacks,
        accelerator,
        tensorboard_logger,
        deterministic,
        logging_dir,
        enable_checkpointing,
        # fitting arguments
        ckpt_path,
        training_loader,
        validation_loader,
        # training loop lightning arguments
        training_loop_lightning_arguments,
    ):

        training_loop_lightning_model = self.training_loop(**training_loop_lightning_arguments)

        self.trainer = Trainer(
            gpus=gpus,
            max_epochs=epoch,
            precision=precision,
            profiler=profiler,
            callbacks=callbacks,
            accelerator=accelerator,
            logger=tensorboard_logger,
            deterministic=deterministic,
            default_root_dir=logging_dir,
            enable_checkpointing=enable_checkpointing,
        )

        self.trainer.fit(
            ckpt_path=ckpt_path,
            model=training_loop_lightning_model,
            train_dataloaders=training_loader if len(training_loader) else None,
            val_dataloaders=validation_loader if len(validation_loader) else None,
        )

    def save(self, yaml_as_dict: Union[None, dict] = None):
        """
        save main components of experiment (model, and yaml file) to be able to be able to reproduce the results

        :param yaml_as_dict: the yaml file that read by the experiment controller
        """
        if self.is_kfold_splitted:
            for i, (trainer_dict, model_state_dict, optimizer_state_dict, scheduler_state_dict) in enumerate(
                zip(
                    *[
                        self.kfolds_trainer_state_dicts,
                        self.kfolds_models_state_dicts,
                        self.kfolds_optimizers_state_dicts,
                        self.kfolds_schedulers_state_dicts,
                    ]
                )
            ):

                self.trainer.__dict__.update(model_state_dict)
                # TODO (Samir): save the model graph, and then save the state dict for each fold
                # self.model.load_state_dict(model_state_dict)
                # self.optimizer.load_state_dict(optimizer_state_dict)
                # self.scheduler.load_state_dict(scheduler_state_dict)

                experiment_kfold_name = join(self.experiment_name, "{}_{}".format(DEFAULT_KFOLD_NAME, str(i)))

                self.__save__(
                    self.model,
                    self.model_inputs,
                    self.optimizer,
                    self.scheduler,
                    yaml_as_dict,
                    self.logging_dir,
                    experiment_kfold_name,
                )
        else:
            self.__save__(
                self.model,
                self.model_inputs,
                self.optimizer,
                self.scheduler,
                yaml_as_dict,
                self.logging_dir,
                self.experiment_name,
            )

    def __save__(self, model, model_inputs, optimizer, scheduler, yaml_as_dict, logging_dir, experiment_name):
        experimenter_writer = ExperimentWriter(logging_dir)
        self.pyq_log_directory = experimenter_writer.write(
            pyq_model=model,
            trainer=self.trainer,
            optimizer=optimizer,
            scheduler=scheduler,
            yaml_as_dict=yaml_as_dict,
            unique_string=experiment_name,
        )

        torch_model = self.get_unparsed_model(deepcopy(model))
        model_name = TORCH_MODEL_FILE_NAME + TORCH_MODEL_EXTENSION
        lightning_dir = join(logging_dir, PYLIGHTNING_LOGS_DIR_NAME, experiment_name)
        ModelWriter(lightning_dir).write(
            torch_model.to(_device(CPU)),
            model_name,
            save_as="graph",
        )

    def clear_logs(self):
        """
        clean the written results for this experiment from the log directory for both pytorch lightning and pyq
        """
        if exists(self.pyq_log_directory):
            shutil.rmtree(self.pyq_log_directory)
        if exists(self.logging_dir):
            shutil.rmtree(self.logging_dir)

    def get_unparsed_model(self, model):
        pyq_model = deepcopy(model)
        return self.unparser.apply(pyq_model)

    def get_profiler(self, log_path):
        """
        :return: pytorch profiler logger to trace back the time of each operation and their statistics
        """
        return PyTorchProfiler(
            dirpath=join(self.logging_dir, PYLIGHTNING_LOGS_DIR_NAME, log_path),
            filename=self.date_time_string,
        )

    def get_tensorboard_logger(self, log_path):
        """
        :return: tensorboard object that can be list with many logger instances
        """
        # TODO (Samir): WandB logger cab be added here and return it as list with `TensorBoardLogger`
        return TensorBoardLogger(
            self.logging_dir,
            name=PYLIGHTNING_LOGS_DIR_NAME,
            version=log_path,
            default_hp_metric=False,
        )
