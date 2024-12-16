from unittest import TestCase, main

from torch.nn import Module

from pyq.datasets.initializer import DataInitializer
from pyq.experiment.classes_builder import InstancesBuilder
from pyq.io.yaml_files import YamlReader


class TestClassBuilderFromYaml(TestCase):
    def setUp(self):
        self.builder = InstancesBuilder()

    def test_build_dataset_from_yaml(self):
        yaml_file_path = "./test_cases/dataset_yaml_test_case.yaml"
        class_as_dict = YamlReader(yaml_file_path).read()
        dataset = self.builder.convert_dictionary_to_instances(class_as_dict)
        self.assertTrue(isinstance(dataset["dataset"], DataInitializer))

    def test_build_wrapper_from_yaml(self):
        yaml_file_path = "./test_cases/wrapper_yaml_test_case.yaml"
        class_as_dict = YamlReader(yaml_file_path).read()
        dataset = self.builder.convert_dictionary_to_instances(class_as_dict)
        self.assertTrue(isinstance(dataset["conv_layer"], Module))


if __name__ == "__main__":
    main()
