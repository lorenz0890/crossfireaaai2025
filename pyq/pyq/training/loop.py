import warnings
from abc import ABC
from typing import Any, List, Optional, Union

from pytorch_lightning import LightningModule
from torch import any, dtype, stack
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import ConstantLR
from torchmetrics.metric import Metric

from pyq.core.wrapper import DEFAULT_WRAPPED_OBJECT_NAME
from pyq.models.wrapper import TorchModelWrapper
from pyq.training.task import Task

DEFAULT_MASK_NAME = "_mask"
DEFAULT_METRIC_NAME = "metric_"
DEFAULT_MAIN_LOSS_NAME = "loss"
DEFAULT_GRAD_NAME = "grad"
DEFAULT_EFFICIENCY_NAME = "efficiency"
DEFAULT_PARAMETER_NAME = "parameters"


class BasicTrainingLoop(LightningModule, ABC):
    def __init__(
        self,
        scheduler: lr_scheduler = None,
        metrics: List[Metric] = None,
        y_transform: Optional[Union[dict, dict]] = None,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler if scheduler else ConstantLR(self.optimizer, factor=1.0)
        self.metrics = metrics if metrics else []
        self.y_transform = y_transform if y_transform else None

    def __logging_learnable_parameters__(self):
        learnable_scaler_param = {}
        # iterating through all parameters
        for name, params in self.named_parameters():
            name = name.replace("." + DEFAULT_WRAPPED_OBJECT_NAME, "")
            learnable_scaler_param.update({name: params})
        return learnable_scaler_param

    def __plot_values__(self, tag_name: str, values_as_dictionary: dict):
        for param_name in values_as_dictionary:
            param_value = values_as_dictionary[param_name]
            if any(param_value.isnan()):
                warnings.warn("{} has NaN gradiant at epoch {}.".format(param_name, self.current_epoch))
                continue
            if param_value.shape.numel() == 1:
                self.log(tag_name + "/" + param_name, values_as_dictionary[param_name], on_step=False, on_epoch=True)
            else:
                self.logger.experiment.add_histogram(param_name, param_value, self.current_epoch)

    def __plot_values_gradient__(self, tag_name: str, values_as_dictionary: dict):
        for name, param in values_as_dictionary:
            if param.grad is None:
                warnings.warn("{} has no gradiant.".format(name))
                continue
            if any(param.grad.isnan()):
                warnings.warn("{} has NaN gradiant at epoch {}.".format(name, self.current_epoch))
                continue
            name = name.replace("." + DEFAULT_WRAPPED_OBJECT_NAME, "") + "." + DEFAULT_GRAD_NAME
            if param.shape.numel() == 1:
                self.log(tag_name + "/" + name, param.grad, on_step=False, on_epoch=True)
            else:
                self.logger.experiment.add_histogram(name, param.grad, self.current_epoch)

    def training_epoch_end(self, outputs):
        for metric in self.metrics:
            metric_name = metric.__class__.__name__.lower()
            metric_average = stack([x[metric_name] for x in outputs]).mean()
            self.log("{}/training".format(metric_name), metric_average, on_step=False, on_epoch=True)

        avg_loss = stack([x[DEFAULT_MAIN_LOSS_NAME] for x in outputs]).mean()
        self.log("step", self.current_epoch)
        self.log(DEFAULT_MAIN_LOSS_NAME + "/training", avg_loss, on_step=False, on_epoch=True)

        learnable_scaler_param = self.__logging_learnable_parameters__()
        self.__plot_values__(DEFAULT_PARAMETER_NAME, learnable_scaler_param)

    def on_after_backward(self):
        if self.current_epoch in self.is_gradients_extracted and self.is_gradients_extracted[self.current_epoch]:
            return
        self.__plot_values_gradient__(DEFAULT_PARAMETER_NAME + "_" + DEFAULT_GRAD_NAME, self.model.named_parameters())
        # set `is_gradients_extracted` to be true for current epoch
        self.is_gradients_extracted.update({self.current_epoch: True})


class TrainingLoop(BasicTrainingLoop):
    """
    Basic class for training loop that uses pytorch lighting and use the `Task` class to manipulate the forward pass,
    and the dataset parameters to select the target features to feed it to the losses
    """

    def __init__(
        self,
        task: Task,
        model: Module,
        losses: List[_Loss],
        optimizer: Optimizer,
        scheduler: lr_scheduler = None,
        metrics: List[Metric] = [],
        y_transform: Optional[Union[dict, dict]] = None,
    ):
        """
        :param model: target model to train, it could be native pytorch model or pytorch geometric model
        :param task: the specific task that needs to be fit into it, e.g. `ImageTask` or `GraphTask`
        :param losses: list of the loss functions that will be summed up to optimize of it
        :param optimizer: the optimizer that tunes the learnable parameters in the `model`
        """
        self.optimizer = optimizer
        super().__init__(scheduler=scheduler, metrics=metrics, y_transform=y_transform)
        self.model = model
        self.task = task
        self.losses_module = losses

        # extract the names form the dataset that will be feed to the model
        self.features_names = self.task.get_input_feature_name()
        [setattr(self, DEFAULT_METRIC_NAME + metric.__class__.__name__.lower(), metric) for metric in self.metrics]

        # debug gradients
        self.is_gradients_extracted = {}

    def __extract_data_point__(self, data):
        """
        determine if there are multiple inputs or not and extract them by index or construct a tuple of input

        :param data: the data which are located in each batch
        :return: the selected data from the batch data to pass it to the model
        """
        if type(self.features_names) is int:
            data_point = data[self.features_names]
        else:
            data_point = (data[input_x] for input_x in self.features_names)
        return data_point

    def forward(self, batch, mode="train"):
        x = self.__extract_data_point__(batch)
        # feed-forward the input data point to the model using task function to be able to pass multiple inputs
        y = self.task.feed_forward_datapoint_to_model(self.model, x)
        if hasattr(batch, mode + DEFAULT_MASK_NAME):
            y = y[getattr(batch, mode + DEFAULT_MASK_NAME)]
        return y

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def __internal_step__(self, y, y_desired, mask):
        # check if mask is passed for graph datasets
        if mask is not None:
            y_desired = y_desired[mask]

        y_desired_transform = y_desired.clone()
        if self.y_transform:
            for key_function_name in self.y_transform:
                y_desired_transform = getattr(y_desired_transform, key_function_name)(
                    self.y_transform[key_function_name]
                )

        # compute the summation of the loss functions
        list_of_losses = [loss(y, y_desired_transform) for loss in self.losses_module]
        sum_of_losses = stack(list_of_losses, dim=0).sum(dim=0).sum(dim=0)
        return_dict = {DEFAULT_MAIN_LOSS_NAME: sum_of_losses}

        for metric in self.metrics:
            metric_name = metric.__class__.__name__.lower()
            metric_class_name = DEFAULT_METRIC_NAME + metric.__class__.__name__.lower()
            device = metric.device if hasattr(metric, "device") else self.device
            metric_value = getattr(self, metric_class_name)(y.to(device), y_desired.to(device))

            return_dict.update({metric_name: metric_value})

        return return_dict

    def training_step(self, batch, batch_idx):
        mode, mask = "train", None
        y = self.forward(batch, mode=mode)
        # extract the desired output from the dataset by its name or index
        y_desired = batch[self.task.get_target_feature_name()]
        if hasattr(batch, mode + DEFAULT_MASK_NAME):
            mask = getattr(batch, mode + DEFAULT_MASK_NAME)
        return self.__internal_step__(y, y_desired, mask)

    def validation_step(self, batch, batch_idx):
        mode, mask = "val", None
        y = self.forward(batch, mode=mode)
        # extract the desired output from the dataset by its name or index
        y_desired = batch[self.task.get_target_feature_name()]
        if hasattr(batch, mode + DEFAULT_MASK_NAME):
            mask = getattr(batch, mode + DEFAULT_MASK_NAME)
        return self.__internal_step__(y, y_desired, mask)

    def training_epoch_end(self, outputs):
        super(TrainingLoop, self).training_epoch_end(outputs)

        if isinstance(self.model, TorchModelWrapper):
            self.model.update_all_callbacks()
            self.__plot_values__(DEFAULT_EFFICIENCY_NAME, self.model.get_callback_dictionary())

        if isinstance(self.model, TorchModelWrapper):
            self.model.update_all_callbacks()
            self.__plot_values__(DEFAULT_EFFICIENCY_NAME, self.model.get_callback_dictionary())

    def validation_epoch_end(self, outputs):
        for metric in self.metrics:
            metric_name = metric.__class__.__name__.lower()
            metric_average = stack([x[metric_name] for x in outputs]).mean()
            self.log("{}/validation".format(metric_name), metric_average, on_step=False, on_epoch=True)

        avg_loss = stack([x[DEFAULT_MAIN_LOSS_NAME] for x in outputs]).mean()

        self.log("step", self.current_epoch)
        self.log(DEFAULT_MAIN_LOSS_NAME + "/validation", avg_loss, on_step=False, on_epoch=True)


class DistilledTrainingLoop(TrainingLoop):
    """
    Training loop to train the student model based on a pre-trained model `teacher_model`.

    source: https://github.com/vrvlive/knowlege-distillation/blob/master/training_module.py#L64
    """

    def __init__(
        self,
        model: Module,
        teacher_model: Module,
        task: Task,
        losses: List[_Loss],
        teacher_losses: List[_Loss],
        loss_coefficient: float,
        optimizer: Optimizer,
        scheduler: lr_scheduler = None,
        metrics: List[Metric] = [],
        y_transform: Optional[Union[dtype, dict]] = None,
    ):
        """
        :param model: student model (tiny model) that has a small number of parameters
        :param teacher_model: pre-trained huge model that has many layers and a large amount of parameters

        see `TrainingLoop` docstring for the definition of the rest of the :params
        """
        super(DistilledTrainingLoop, self).__init__(
            model=model,
            task=task,
            losses=losses,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            y_transform=y_transform,
        )
        self.teacher_model = teacher_model
        self.teacher_losses = teacher_losses
        self.loss_coefficient = loss_coefficient
        self.teacher_task = task.__class__(task_name=task.task_name, model=teacher_model, dataset=task.dataset)

        if not self.teacher_task.is_dataset_compatible_with_model():
            raise ValueError(
                "{} dataset is not compatible with the model {}".format(
                    task.dataset.__class__.__name__, teacher_model.__class__.__name__
                )
            )

        # freeze the learnable parameters for the teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def __internal_step__(self, y, y_teacher, y_desired, mask):
        # check if mask is passed for graph datasets
        if mask is not None:
            y_desired = y_desired[mask]
            y_teacher = y_teacher[mask]

        y_desired_transform = y_desired.clone()
        if self.y_transform:
            for key_function_name in self.y_transform:
                y_desired_transform = getattr(y_desired_transform, key_function_name)(
                    self.y_transform[key_function_name]
                )

        # compute the summation of the loss functions
        list_of_losses = [loss(y, y_desired_transform) for loss in self.losses_module]

        sum_of_losses = stack(list_of_losses, dim=0).sum(dim=0).sum(dim=0) if list_of_losses else 0

        # compute the summation of the loss functions
        list_of_teacher_losses = [loss(y.log_softmax(dim=-1), y_teacher) for loss in self.teacher_losses]
        sum_of_teacher_losses = stack(list_of_teacher_losses, dim=0).sum(dim=0).sum(dim=0)

        total_losses = self.loss_coefficient * sum_of_losses + (1 - self.loss_coefficient) * sum_of_teacher_losses

        return_dict = {DEFAULT_MAIN_LOSS_NAME: total_losses}

        for metric in self.metrics:
            metric_name = metric.__class__.__name__.lower()
            metric_class_name = DEFAULT_METRIC_NAME + metric.__class__.__name__.lower()
            device = metric.device if hasattr(metric, "device") else self.device
            metric_value = getattr(self, metric_class_name)(y.to(device), y_desired.to(device))

            return_dict.update({metric_name: metric_value})

        return return_dict

    def training_step(self, batch, batch_idx):
        mode, mask = "train", None
        x = self.__extract_data_point__(batch)
        y_hat_student = self.forward(batch, mode=mode)
        y_hat_teacher = self.task.feed_forward_datapoint_to_model(self.teacher_model, x).log_softmax(dim=-1)
        y_desired = batch[self.task.get_target_feature_name()]

        if hasattr(batch, mode + DEFAULT_MASK_NAME):
            mask = getattr(batch, mode + DEFAULT_MASK_NAME)
        return self.__internal_step__(y_hat_student, y_hat_teacher, y_desired, mask)

    def validation_step(self, batch, batch_idx):
        mode, mask = "val", None
        x = self.__extract_data_point__(batch)
        y_hat_student = self.forward(batch, mode=mode)
        y_hat_teacher = self.teacher_task.feed_forward_datapoint_to_model(self.teacher_model, x).log_softmax(dim=-1)
        y_desired = batch[self.task.get_target_feature_name()]

        if hasattr(batch, mode + DEFAULT_MASK_NAME):
            mask = getattr(batch, mode + DEFAULT_MASK_NAME)
        return self.__internal_step__(y_hat_student, y_hat_teacher, y_desired, mask)
