from collections import KeysView
from unittest import TestCase, main, skip

from pyq.datasets.initializer import DatasetCharacteristicExtractor
from pyq.experiment.classes_builder import InstancesBuilder


class TestDatasetInitializer(TestCase):
    def setUp(self):
        dataset_initializer_class_name = "DataInitializer"
        self.builder = InstancesBuilder()

        self.mnist_test_case = {
            "class_name": dataset_initializer_class_name,
            "dataset_name": "mnist",
            "transform": {"class_name": "ToTensor"},
        }

        self.planetoid_test_case = {
            "class_name": dataset_initializer_class_name,
            "dataset_name": "planetoid",
            "name": "Cora",
        }

        self.amazon_test_case = {
            "class_name": dataset_initializer_class_name,
            "dataset_name": "amazon",
            "name": "Computers",
        }

        self.tu_dataset_test_case = {
            "class_name": dataset_initializer_class_name,
            "dataset_name": "TUDataset",
            "name": "AIDS",
        }

        self.ogb_arxiv_test_case = {
            "class_name": dataset_initializer_class_name,
            "dataset_name": "ogbn-arxiv",
            "name": "ogbn-arxiv",
        }

        # self.ogb_ddi_test_case = {
        #     "class_name": dataset_initializer_class_name,
        #     "dataset_name": "ogbl-ddi",
        #     "name": "ogbl-ddi",
        # }

    @skip("skip downloading of the dataset")
    def test_mnist_dataset(self):
        mnist_dataset = self.builder.construct_instance(self.mnist_test_case).get_train_test_set()
        self.assertTrue(mnist_dataset)

    @skip("skip downloading of the dataset")
    def test_cora_dataset(self):
        cora_dataset = self.builder.construct_instance(self.planetoid_test_case).get_train_test_set()
        self.assertTrue(cora_dataset)

    @skip("skip downloading of the dataset")
    def test_amazon_computers_dataset(self):
        amazon_computers_dataset = self.builder.construct_instance(self.amazon_test_case).get_train_test_set()
        self.assertTrue(amazon_computers_dataset)

    @skip("skip downloading of the dataset")
    def test_aids_dataset(self):
        aids_dataset = self.builder.construct_instance(self.tu_dataset_test_case).get_train_test_set()
        self.assertTrue(aids_dataset)

    # TODO (Samir): Add node prediction and handle it's splits
    @skip("skip downloading of the dataset")
    def test_ogb_arxiv_dataset(self):
        ogb_arxiv_dataset = self.builder.construct_instance(self.ogb_arxiv_test_case).get_train_test_set()
        self.assertTrue(ogb_arxiv_dataset)

    # TODO (Samir): Add link prediction and handle it's splits
    # def test_ogb_ddi_dataset(self):
    #     ogb_ddi_dataset = self.builder.construct_instance(self.ogb_ddi_test_case).get_train_test_set()
    #     self.assertTrue(ogb_ddi_dataset)


class DatasetFeatureExtractor(TestDatasetInitializer):
    def setUp(self):
        super().setUp()

    @skip("skip downloading of the dataset")
    def test_mnist_dataset(self):
        train_dataset, test_dataset = self.builder.construct_instance(self.mnist_test_case).get_train_test_set()
        dataset_characteristic = DatasetCharacteristicExtractor(train_dataset)
        names = dataset_characteristic.extract_data_names()
        self.assertTrue(isinstance(names, KeysView))

    @skip("skip downloading of the dataset")
    def test_cora_dataset(self):
        train_dataset, test_dataset = self.builder.construct_instance(self.planetoid_test_case).get_train_test_set()
        dataset_characteristic = DatasetCharacteristicExtractor(train_dataset)
        names = dataset_characteristic.extract_data_names()
        self.assertTrue(isinstance(names, KeysView))

    @skip("skip downloading of the dataset")
    def test_amazon_computers_dataset(self):
        train_dataset, test_dataset = self.builder.construct_instance(self.amazon_test_case).get_train_test_set()
        dataset_characteristic = DatasetCharacteristicExtractor(train_dataset)
        names = dataset_characteristic.extract_data_names()
        self.assertTrue(isinstance(names, KeysView))

    @skip("skip downloading of the dataset")
    def test_aids_dataset(self):
        train_dataset, test_dataset = self.builder.construct_instance(self.tu_dataset_test_case).get_train_test_set()
        dataset_characteristic = DatasetCharacteristicExtractor(train_dataset)
        names = dataset_characteristic.extract_data_names()
        self.assertTrue(isinstance(names, KeysView))

    @skip("skip downloading of the dataset")
    def test_ogb_arxiv_dataset(self):
        train_dataset, test_dataset = self.builder.construct_instance(self.ogb_arxiv_test_case).get_train_test_set()
        dataset_characteristic = DatasetCharacteristicExtractor(train_dataset)
        names = dataset_characteristic.extract_data_names()
        self.assertTrue(isinstance(names, KeysView))

    # def test_ogb_ddi_dataset(self):
    #     train_dataset, test_dataset = self.builder.construct_instance(self.ogb_ddi_test_case).get_train_test_set()
    #     dataset_characteristic = DatasetCharacteristicExtractor(train_dataset)
    #     names = dataset_characteristic.extract_data_names()
    #     self.assertTrue(isinstance(names, KeysView))


if __name__ == "__main__":
    main()
