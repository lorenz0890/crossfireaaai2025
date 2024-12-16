import warnings
from abc import ABC, abstractmethod
from operator import itemgetter

import torch
import torch_geometric
import torchvision

from pyq.datasets.initializer import Y_NAME, DatasetCharacteristicExtractor
from pyq.experiment.function_builder import DEFAULT_FORWARD_METHOD_NAME, get_function_argument_names_and_values

DEFAULT_ARGS_METHOD_NAME = "args"
DEFAULT_KWARGS_METHOD_NAME = "kwargs"


class Task(ABC):
    """
    A task that suppose to check the pip-line components before the training loop and check inputs compatibility with
    both the training loop and the model to support image and graph classification.
    TODO (Samir): make it more generic by decomposing this parent class into many other tasks to support embedding and
                  segmentation and generative models and their pipe-lines
    """

    def __init__(self, task_name, dataset=None, model=None):
        """
        :param task_name: name of the task, for now, there is no use for this variable
        :param dataset: dataset that will be passed to the training loop
        """
        self.task_name = task_name
        self.model = model
        self.dataset = dataset
        self._compatible_datasets_with_task = None

    def get_common_attribute_dataset_model(self, model):
        """
        Extract the common names between the dataset feature's names and the input parameter's name

        :param model: torch model that has `forward` and a primary input called `x`
        :return: set of common names between the model `forward` function and names of features from the dataset
        """
        dataset_features = DatasetCharacteristicExtractor(self.dataset)
        dataset_point = dataset_features.sample_data_point_as_dict()
        model_input_keys = get_function_argument_names_and_values(model, DEFAULT_FORWARD_METHOD_NAME, False).keys()
        common_keys = set(model_input_keys).intersection(dataset_point.keys())
        sorted_common_keys = sorted(common_keys & set(model_input_keys), key=list(model_input_keys).index)
        return sorted_common_keys

    def get_model_inputs(self):
        dataset_features = DatasetCharacteristicExtractor(self.dataset)
        dataset_point = dataset_features.sample_data_point_as_dict()
        common_keys = self.get_common_attribute_dataset_model(self.model)
        model_input = itemgetter(*common_keys)(dataset_point)
        return model_input

    def is_dataset_compatible_with_model(self):
        """
        check dataset compatibility with the model by extracting a data point from it and feed-forward it to the model

        :return: True if the output from the model is tensor, otherwise false
        """
        model_input = self.get_model_inputs()
        y = self.feed_forward_datapoint_to_model(self.model, model_input)
        return torch.is_tensor(y)

    def is_dataset_compatible_with_task(self):
        """
        check if the dataset which is provided is an instance from the libraries that support the target task

        :return: True if the dataset is inherent from the family of the dataset that supports the task
        """
        return isinstance(self.dataset, tuple(self._compatible_datasets_with_task))

    def set_model(self, model):
        """
        set the model after constructing the task class.
        :param model: target model to set it in the current scope
        """
        if self.model:
            warnings.warn(
                "model is already set to be {}, it will replace with model {}".format(
                    self.model.__class__.__name__, model.__class__.__name__
                )
            )
        self.model = model

    def set_dataset(self, dataset):
        """
        set the dataset after constructing the task class.
        :param dataset: target dataset to set it in the current scope
        """
        if self.dataset:
            warnings.warn(
                "dataset is already set to be {}, it will replace with dataset {}".format(
                    self.dataset.__class__.__name__, dataset.__class__.__name__
                )
            )
        self.dataset = dataset

    @abstractmethod
    def feed_forward_datapoint_to_model(self, model, datapoint):
        pass

    @abstractmethod
    def get_target_feature_name(self):
        pass

    @abstractmethod
    def get_input_feature_name(self):
        pass

    @abstractmethod
    def get_dataloader_class(self):
        pass


class ImageTask(Task):
    """
    Class that supports image task as image classification, by checking that the provided dataset is from torch-vision
    """

    def __init__(self, task_name, dataset, model):
        super().__init__(task_name, dataset, model)
        self._compatible_datasets_with_task = [torchvision.datasets.VisionDataset]

    def feed_forward_datapoint_to_model(self, model, dataset_point):
        return model(dataset_point)

    def get_input_feature_name(self):
        dataset_features = DatasetCharacteristicExtractor(self.dataset)
        if dataset_features.has_x_feature():
            return 0
        raise ValueError("dataset {} has no x feature, index 0".format(self.dataset.__class__.__name__))

    def get_target_feature_name(self):
        dataset_features = DatasetCharacteristicExtractor(self.dataset)
        if dataset_features.has_y_feature():
            return 1
        raise ValueError("dataset {} has no y feature, index 1".format(self.dataset.__class__.__name__))

    def get_dataloader_class(self):
        return torch.utils.data.DataLoader


class GraphTask(Task):
    """
    Class that supports graph task as graph classification, node embedding by checking that the provided dataset is from
    pytorch-geometric dataset `InMemoryDataset`
    """

    def __init__(self, task_name, dataset, model):
        super().__init__(task_name, dataset, model)
        self._compatible_datasets_with_task = [torch_geometric.data.InMemoryDataset, torch_geometric.data.Data]

    def feed_forward_datapoint_to_model(self, model, dataset_point):
        return model(*dataset_point)

    def get_input_feature_name(self):
        input_feature_name = list(
            get_function_argument_names_and_values(self.model, DEFAULT_FORWARD_METHOD_NAME, False).keys()
        )
        if DEFAULT_ARGS_METHOD_NAME in input_feature_name:
            input_feature_name.remove(DEFAULT_ARGS_METHOD_NAME)
        if DEFAULT_KWARGS_METHOD_NAME in input_feature_name:
            input_feature_name.remove(DEFAULT_KWARGS_METHOD_NAME)
        return input_feature_name

    def get_target_feature_name(self):
        dataset_features = DatasetCharacteristicExtractor(self.dataset)
        if dataset_features.has_y_feature():
            return Y_NAME
        raise ValueError("dataset {} has no attribute named `y`".format(self.dataset.__class__.__name__))

    def get_dataloader_class(self):
        return torch_geometric.loader.DataLoader
