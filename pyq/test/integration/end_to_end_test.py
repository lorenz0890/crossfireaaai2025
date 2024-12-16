from unittest import TestCase, main

from torch import device
from torch.cuda import is_available

from pyq.experiment.controller import ExperimentController
from pyq.io.base import InputOutput


class TestEndToEnd(TestCase):
    def setUp(self):
        test_log_dir = "./datasets/"
        InputOutput.create_dir(test_log_dir)

        device_ = device("cuda:0") if is_available() else device("cpu")
        self.experiment_controller = ExperimentController(device_, test_log_dir)

    def test_gcn_ogbg_quant_end_to_end(self):
        config_path = "./test_cases/end_to_end_graph_quant_test.yaml"
        self.experiment_controller.setup_experiment_parameters(yaml_file_path=config_path)
        self.experiment_controller.dispatch_experiment()
        self.experiment_controller.finalize_experiment()
        self.experiment_controller.experiment_dispatcher.clear_logs()

    def test_alexnet_mnist_quant_end_to_end(self):
        config_path = "./test_cases/end_to_end_vision_quant_test.yaml"
        self.experiment_controller.setup_experiment_parameters(yaml_file_path=config_path)
        self.experiment_controller.dispatch_experiment()
        self.experiment_controller.finalize_experiment()
        self.experiment_controller.experiment_dispatcher.clear_logs()


if __name__ == "__main__":
    main()
