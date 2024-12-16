import copy
import os
import warnings
from typing import List

import pandas
import torch_geometric
import torchvision
import torchvision.datasets
from ogb import graphproppred, nodeproppred
from torch import Tensor, bool, rand, randperm, squeeze, stack, utils, zeros
from torch_geometric.data import Batch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import index_to_mask

from pyq.datasets.splitter import TorchVisionDatasetSplitter
from pyq.experiment.default_values import X_NAME, Y_NAME
from pyq.paths import PyqPath

DATASET_SUPPORT_INDEXING = (
    # graphproppred.PygGraphPropPredDataset,
    # nodeproppred.PygNodePropPredDataset,
    # linkproppred.PygLinkPropPredDataset,
    Planetoid,
)
NONE_NAMED_FEATURE = "non_registered_feature"
DEFAULT_TRANSFORMATION_NAME = "transform"
TRAIN_TRANSFORMATION_NAME = "train_transform"
TEST_TRANSFORMATION_NAME = "test_transform"
TRAIN_NAME = "train"
VAL_NAME = "val"
VALID_NAME = "valid"
TEST_NAME = "test"
MASK_NAME = "mask"
DEFAULT_TRAIN_MASK_NAME = "_".join([TRAIN_NAME, MASK_NAME])
DEFAULT_VAL_MASK_NAME = "_".join([VAL_NAME, MASK_NAME])
DEFAULT_TEST_MASK_NAME = "_".join([TEST_NAME, MASK_NAME])


def find_name_in_list(name: str, target_list: List[str]):
    """
    search for the string name in the provided list, such that the lower case should match and return the
    original name from the list

    :param name: main name that the function should return its similar one from the list
    :param target_list: the list to search into it
    :return: the first occurrence of the `name` from `target_list`
    """

    for local_name in target_list:
        if local_name.lower() == name.lower():
            return local_name


def generate_masks(y: Tensor, num_splits: int = 1, train_per_class: int = 1024, val_per_class: int = 1024):
    # source: https://github.com/rusty1s/pyg_autoscale/blob/master/torch_geometric_autoscale/utils.py#L38
    num_classes = int(y.max()) + 1
    train_mask = zeros(y.size(0), num_splits, dtype=bool)
    val_mask = zeros(y.size(0), num_splits, dtype=bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        perm = stack([randperm(idx.size(0)) for _ in range(num_splits)], dim=1)
        idx = idx[perm]
        train_idx = idx[:train_per_class]
        train_mask.scatter_(0, train_idx, True)
        val_idx = idx[train_per_class: train_per_class + val_per_class]
        val_mask.scatter_(0, val_idx, True)

    test_mask = ~(train_mask | val_mask)
    return squeeze(train_mask), squeeze(val_mask), squeeze(test_mask)


class DataInitializer:
    """
    Dataset prepare class to download, parse input, and return the target dataset based on its name and arguments
    that passed to it.
    """

    def __init__(
        self,
        dataset_name: str,
        split_ratio: float = 0.7,
        root: str = "{}/datasets/".format(os.getcwd()),
        train_on_all_sets: bool = False,
        *args,
        **kwargs
    ):
        """
        :param dataset_name: the family name of the dataset. e.g. planetoid, amazon, or tudataset
        :param root: path to download the dataset into it, by default, it would be inside the repository
        :param args, kwargs: all the arguments that need to pass to the dataset loader, e.g., transform
        """
        self.root = root
        self.split_ratio = split_ratio
        self.dataset_name = dataset_name.lower()
        self.train_on_all_sets = train_on_all_sets
        self.args, self.kwargs = args, kwargs
        [setattr(self, k, kwargs[k]) for k in kwargs]

        # split the training transform and testing transform into two separate kwargs
        self.train_kwargs, self.test_kwargs = copy.deepcopy(kwargs), copy.deepcopy(kwargs)
        if TEST_TRANSFORMATION_NAME in self.train_kwargs:
            self.train_kwargs.pop(TEST_TRANSFORMATION_NAME)
        if TRAIN_TRANSFORMATION_NAME in self.test_kwargs:
            self.test_kwargs.pop(TRAIN_TRANSFORMATION_NAME)

        if TRAIN_TRANSFORMATION_NAME in self.train_kwargs:
            self.train_kwargs[DEFAULT_TRANSFORMATION_NAME] = self.train_kwargs.pop(TRAIN_TRANSFORMATION_NAME)
        if TEST_TRANSFORMATION_NAME in self.test_kwargs:
            self.test_kwargs[DEFAULT_TRANSFORMATION_NAME] = self.test_kwargs.pop(TEST_TRANSFORMATION_NAME)

        ogb_graph_path = PyqPath.get_class_path(graphproppred.PygGraphPropPredDataset)
        ogb_node_path = PyqPath.get_class_path(nodeproppred.PygNodePropPredDataset)
        # ogb_link_path = PyqPath.get_class_path(linkproppred.PygLinkPropPredDataset)

        self.ogb_graph_datasets = pandas.read_csv(os.path.dirname(ogb_graph_path) + "/master.csv", index_col=0).columns
        self.ogb_node_datasets = pandas.read_csv(os.path.dirname(ogb_node_path) + "/master.csv", index_col=0).columns
        # self.ogb_link_datasets = pandas.read_csv(os.path.dirname(ogb_link_path) + "/master.csv", index_col=0).columns

        self.torch_datasets = list((map(lambda x: x.lower(), dir(torchvision.datasets))))
        self.torch_geometric_datasets = list((map(lambda x: x.lower(), dir(torch_geometric.datasets))))
        self.ogb_graph_datasets = list((map(lambda x: x.lower(), self.ogb_graph_datasets)))
        self.ogb_node_datasets = list((map(lambda x: x.lower(), self.ogb_node_datasets)))
        # self.ogb_link_datasets = list((map(lambda x: x.lower(), self.ogb_link_datasets)))

        self._available_datasets = (
            self.torch_datasets
            + self.torch_geometric_datasets
            + self.ogb_graph_datasets
            + self.ogb_node_datasets
            # + self.ogb_link_datasets
        )

        if not self.__is_available_dataset__():
            raise ValueError(
                "incorrect `dataset_name`: {}, list of available dataset {}".format(
                    self.dataset_name, self._available_datasets
                )
            )

    def __is_available_dataset__(self):
        """
        check if the dataset is available in `torchvision` or `torch_geometric`

        :return: True if the dataset is available, otherwise False
        """
        return self.dataset_name in self._available_datasets

    def __exact_dataset_name__(self):
        """
        :return: the real dataset name which is written in the libraries
        """
        torchvision_name = find_name_in_list(self.dataset_name, dir(torchvision.datasets))
        torch_geometric_name = find_name_in_list(self.dataset_name, dir(torch_geometric.datasets))
        ogb_graph_name = find_name_in_list(self.dataset_name, self.ogb_graph_datasets)
        ogb_node_name = find_name_in_list(self.dataset_name, self.ogb_node_datasets)
        # TODO (Samir): uncomment to add link tasks
        # ogb_link_name = find_name_in_list(self.dataset_name, self.ogb_link_datasets)
        ogb_link_name = None

        return torchvision_name, torch_geometric_name, ogb_graph_name, ogb_node_name, ogb_link_name

    @property
    def full_dataset_name(self):
        return self.dataset_name + (
            "_" + self.name.lower() if hasattr(self, "name") and self.name != self.dataset_name else ""
        )

    def __get_torch_vision_train_test_set__(self, dataset_name):
        if self.dataset_name == torchvision.datasets.ImageNet.__name__.lower():
            train_set = getattr(torchvision.datasets, dataset_name)(
                root=self.root, split=TRAIN_NAME, *self.args, **self.train_kwargs
            )
            test_set = getattr(torchvision.datasets, dataset_name)(
                root=self.root, split=VAL_NAME, *self.args, **self.test_kwargs
            )
            return train_set, test_set
        if self.dataset_name == torchvision.datasets.ImageFolder.__name__.lower():
            train_set = getattr(torchvision.datasets, dataset_name)(
                root=self.root + "/" + TRAIN_NAME, *self.args, **self.train_kwargs
            )
            test_set = getattr(torchvision.datasets, dataset_name)(
                root=self.root + "/" + VAL_NAME, *self.args, **self.test_kwargs
            )
            return train_set, test_set
        else:
            train_set = getattr(torchvision.datasets, dataset_name)(
                root=self.root, download=True, *self.args, **self.train_kwargs
            )
            test_set = getattr(torchvision.datasets, dataset_name)(
                root=self.root, download=True, *self.args, **self.test_kwargs
            )
            train_splitter = TorchVisionDatasetSplitter(train_set)
            test_splitter = TorchVisionDatasetSplitter(test_set)
            return train_splitter.get_training_set(), test_splitter.get_test_set()

    def __get_torch_geometric_train_test_set__(self, dataset_name):
        source_set = getattr(torch_geometric.datasets, dataset_name)(root=self.root, *self.args, **self.kwargs)
        train_data, val_data = copy.deepcopy(source_set), copy.deepcopy(source_set)

        if self.train_on_all_sets:
            if hasattr(train_data[0], DEFAULT_TEST_MASK_NAME):
                for train_set in train_data:
                    train_set.train_mask.fill_(True)
            else:
                train_data.train_mask.fill_(True)

        if len(source_set) == 1:
            # if the returned dataset is a single graph return it to be training and testing sets
            source_set = source_set[0]
            if not (hasattr(source_set, DEFAULT_TRAIN_MASK_NAME) and hasattr(source_set, DEFAULT_TEST_MASK_NAME)):
                train_mask, val_mask, test_mask = generate_masks(y=source_set.y, num_splits=1)
                setattr(source_set, DEFAULT_TRAIN_MASK_NAME, train_mask)
                setattr(source_set, DEFAULT_VAL_MASK_NAME, val_mask)
                setattr(source_set, DEFAULT_TEST_MASK_NAME, test_mask)

            train_data[0].y = train_data[0].y[getattr(source_set, DEFAULT_TRAIN_MASK_NAME)]
            val_data[0].y = val_data[0].y[getattr(source_set, DEFAULT_VAL_MASK_NAME)]
        else:
            if hasattr(source_set[0], DEFAULT_TRAIN_MASK_NAME) and hasattr(source_set[0], DEFAULT_TEST_MASK_NAME):
                for graph_idx in range(len(source_set)):
                    train_data[graph_idx].y = train_data[graph_idx].y[train_data[graph_idx].train_mask]
                    val_data[graph_idx].y = val_data[graph_idx].y[val_data[graph_idx].test_mask]
            else:
                random_mask = rand(len(source_set)) < self.split_ratio
                train_data, val_data = train_data[random_mask], val_data[~random_mask]

                warnings.warn(
                    "the graphs will be split automatically with a ratio equal to {}%. To tune it, "
                    "pass split_ratio to {}".format(
                        (1 - len(val_data) / len(train_data)) * 100, self.__class__.__name__
                    )
                )
        return train_data, val_data

    def get_train_test_set(self):
        """
        Download the dataset to the `root` directory and parse to target attribute as `transform`, `split` to the
        dataset library, this function wouldn't raise any exception science that function `_is_available_dataset` is
        called in the constructor

        :return: tuple holds two instances of the dataset as training and testing sets.
        """
        (
            vision_dataset_name,
            geometric_dataset_name,
            ogb_graph_dataset_name,
            ogb_node_dataset_name,
            ogb_link_dataset_name,
        ) = self.__exact_dataset_name__()

        if vision_dataset_name:
            training_dataset, validation_dataset = self.__get_torch_vision_train_test_set__(vision_dataset_name)
            return training_dataset, validation_dataset

        if geometric_dataset_name:
            training_dataset, validation_dataset = self.__get_torch_geometric_train_test_set__(geometric_dataset_name)
            return training_dataset, validation_dataset

        if ogb_graph_dataset_name:
            dataset = graphproppred.PygGraphPropPredDataset(root=self.root, *self.args, **self.kwargs)
            split_idx = dataset.get_idx_split()
            training_dataset = dataset[split_idx[TRAIN_NAME]]
            validation_dataset = dataset[split_idx[VALID_NAME]]
            return training_dataset, validation_dataset

        if ogb_node_dataset_name:
            dataset = nodeproppred.PygNodePropPredDataset(root=self.root, *self.args, **self.kwargs)

            if self.train_on_all_sets:
                training_dataset, validation_dataset = copy.deepcopy(dataset), copy.deepcopy(dataset)
                return training_dataset, validation_dataset

            split_idx = dataset.get_idx_split()
            training_dataset, validation_dataset = [], []
            for graph_idx in range(len(dataset)):
                training_dict = dataset[graph_idx].to_dict().copy()
                validation_dict = dataset[graph_idx].to_dict().copy()
                for key in [DEFAULT_TRAIN_MASK_NAME, DEFAULT_VAL_MASK_NAME, DEFAULT_TEST_MASK_NAME]:
                    data_dim = dataset[graph_idx].y.shape[0]
                    training_dict.update({key: index_to_mask(split_idx[TRAIN_NAME], data_dim)})
                    validation_dict.update({key: index_to_mask(split_idx[VALID_NAME], data_dim)})

                training_dataset.append(dataset[graph_idx].from_dict(training_dict))
                validation_dataset.append(dataset[graph_idx].from_dict(validation_dict))
            return Batch.from_data_list(training_dataset), Batch.from_data_list(validation_dataset)

        # TODO (Samir): Add link prediction and handle it's splits
        # if ogb_link_dataset_name:
        #     dataset = linkproppred.PygLinkPropPredDataset(root=self.root, *self.args, **self.kwargs)
        #     training_dataset, validation_dataset = _ogb_split_train_test_set(dataset)
        #     return training_dataset, validation_dataset


class DatasetCharacteristicExtractor:
    """
    Extract some primary attributes for the dataset to automate the process of checking if the dataset is compatible
    for the model or not, based on the number of the attribute to feed to the model and inputs size
    """

    def __init__(self, dataset):
        """
        :param dataset: a dataset which is going to train the model on its data points
        """
        if isinstance(dataset, DATASET_SUPPORT_INDEXING):
            self.data_point = dataset.get(0)
        elif isinstance(dataset, utils.data.DataLoader):
            self.data_point = next(dataset.__iter__())
        elif isinstance(dataset, torchvision.datasets.VisionDataset):
            loader = utils.data.DataLoader(dataset, shuffle=False)
            self.data_point = next(loader.__iter__())
        elif isinstance(dataset, torch_geometric.data.Data):
            self.data_point = dataset
        elif isinstance(dataset, torch_geometric.data.InMemoryDataset):
            # TODO (Samir):  sampling: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
            loader = torch_geometric.loader.DataLoader(dataset, shuffle=False)
            self.data_point = next(loader.__iter__())
        else:
            raise ValueError("{} unsupported dataset".format(dataset.__class__.__name__))

        # apply transformation if it exists for graph task
        if (
            hasattr(dataset, "transform")
            and dataset.transform
            and not isinstance(dataset, torchvision.datasets.VisionDataset)
        ):
            self.data_point = dataset.transform(self.data_point)

    def sample_data_point_from_dataset(self):
        """
        :return: single (could be random also) data point from the dataset
        """
        return self.data_point

    def sample_data_point_as_dict(self) -> dict:
        """
        :return: dictionary that hold in its key the names of the feature and in its values the data of these features
        """
        if isinstance(self.data_point, list):
            if len(self.data_point) == 1:
                return {X_NAME: self.data_point[0]}
            if len(self.data_point) == 2:
                return {X_NAME: self.data_point[0], Y_NAME: self.data_point[1]}
            else:
                return {NONE_NAMED_FEATURE + "_{}".format(i): data for i, data in enumerate(self.data_point)}
        data_point_as_dict = {k: getattr(self.data_point, k) for k in dir(self.data_point)}
        data_point_as_dict.update({k: self.data_point[k] for k in self.data_point.keys})
        return data_point_as_dict

    def extract_data_names(self):
        """
        :return: return the names of the features from the data point, if it does not exist
                 `non_registered_feature_i` will assign to each feature
        """
        return self.sample_data_point_as_dict().keys()

    def extract_data_shapes(self):
        """
        :return: dictionary that hold in its key the names of the feature and in its values the shape of the data
        """
        data_as_dict = self.sample_data_point_as_dict()
        return {k: data_as_dict[k].shape for k in data_as_dict}

    def has_x_feature(self):
        """
        :return: Ture if the dataset has attributed with name `x`, otherwise False
        """
        return X_NAME in self.extract_data_names()

    def has_y_feature(self):
        """
        :return: Ture if the dataset has attributed with name `y`, otherwise False
        """
        return Y_NAME in self.extract_data_names()
