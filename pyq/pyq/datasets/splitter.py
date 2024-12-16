import inspect
import warnings
from abc import ABC

from numpy import array
from sklearn.model_selection import KFold, StratifiedKFold
from torch import bool, from_numpy, long, ones, zeros
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

from pyq.experiment.function_builder import get_function_argument_names_and_values


class DatasetSplitter(ABC):
    """
    Abstract class to get the training and testing dataset
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def get_training_set(self):
        raise NotImplementedError(
            "{} didn't override the function `{}`".format(
                self.__class__.__name__, inspect.currentframe().f_code.co_name
            )
        )

    def get_test_set(self):
        raise NotImplementedError(
            "{} didn't override the function `{}`".format(
                self.__class__.__name__, inspect.currentframe().f_code.co_name
            )
        )


class DataKFold(ABC):
    """
    Abstract class to compute the indices for the k-fold
    """

    def __init__(self, n_splits: int):
        assert n_splits > 2, "number of splits for kfold should be more than two."
        self.n_splits = n_splits

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_indices_list(self):
        raise NotImplementedError(
            "{} didn't override the function `{}`".format(
                self.__class__.__name__, inspect.currentframe().f_code.co_name
            )
        )


class TorchVisionDatasetSplitter(DatasetSplitter):
    """
    Dataset splitter to split the torch vision library dataset
    """

    def __init__(self, dataset):
        """
        :param dataset: dataset instance which could be only training or testing dataset
        """
        super(TorchVisionDatasetSplitter, self).__init__(dataset)
        self._dataset_provided_arguments = get_function_argument_names_and_values(dataset)

    def get_training_set(self):
        if isinstance(self.dataset, (MNIST, CIFAR10, CIFAR100)):
            self._dataset_provided_arguments["train"] = True
            return self.dataset.__class__(**self._dataset_provided_arguments)

        raise ValueError(
            "{}, unsupported dataset to split using {}.".format(
                self.dataset.__class__.__name__, self.__class__.__name__
            )
        )

    def get_test_set(self):
        if isinstance(self.dataset, (MNIST, CIFAR10, CIFAR100)):
            self._dataset_provided_arguments["train"] = False
            return self.dataset.__class__(**self._dataset_provided_arguments)

        raise ValueError(
            "{}, unsupported dataset to split using {}.".format(
                self.dataset.__class__.__name__, self.__class__.__name__
            )
        )


class TorchVisionDataKFold(DataKFold):
    """
    Split the computer vision datasets into `n_splits` with their indices
    """

    def __init__(self, n_splits):
        super(TorchVisionDataKFold, self).__init__(n_splits)

    def get_indices_list(self):
        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        train_indices, test_indices = array([*kfold.split(self.dataset.data, self.dataset.targets)], dtype=list).T
        return train_indices, test_indices


class GraphDataKFold(DataKFold):
    """
    Split the graph's data into `n_splits` with their indices
    """

    def __init__(self, n_splits):
        super(GraphDataKFold, self).__init__(n_splits)

    def get_indices_list(self):
        number_data_points = len(self.dataset.data)
        number_y_points = len(self.dataset.data.y)

        if number_data_points == number_y_points:
            kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            splits = kfold.split(zeros(len(self.dataset)), self.dataset.data.y)
        else:
            warnings.warn("number of data points is not equal to the target points.")
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            splits = kfold.split(zeros(len(self.dataset)))

        test_indices, train_indices = [], []
        for _, idx in splits:
            test_indices.append(from_numpy(idx).to(long))

        for i in range(self.n_splits):
            train_mask = ones(len(self.dataset), dtype=bool)
            train_mask[test_indices[i]] = 0
            train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

        return train_indices, test_indices
