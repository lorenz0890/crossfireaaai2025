from unittest import TestCase, main

from torchvision import transforms

from pyq.datasets.initializer import DataInitializer
from pyq.training.task import GraphTask, ImageTask
from pyq.utils import GCN, ConvNet


class TestTasks(TestCase):
    def setUp(self):
        pass

    def test_image_task(self):
        data_initializer = DataInitializer(dataset_name="MNIST", transform=transforms.ToTensor())
        training_dataset, validation_dataset = data_initializer.get_train_test_set()
        mnist_model = ConvNet(1, 10)
        task = ImageTask("classification", training_dataset, mnist_model)
        self.assertTrue(task.is_dataset_compatible_with_task())
        self.assertTrue(task.is_dataset_compatible_with_model())

    def test_graph_task(self):
        data_initializer = DataInitializer(dataset_name="Planetoid", name="Cora")
        training_dataset, validation_dataset = data_initializer.get_train_test_set()
        gcn_model = GCN(in_channels=training_dataset.num_features, out_channels=training_dataset.num_classes)
        task = GraphTask("classification", training_dataset, gcn_model)
        self.assertTrue(task.is_dataset_compatible_with_task())
        self.assertTrue(task.is_dataset_compatible_with_model())


if __name__ == "__main__":
    main()
