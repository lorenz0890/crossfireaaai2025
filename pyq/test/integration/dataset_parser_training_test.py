from unittest import TestCase, main, skip

import torch
from pytorch_lightning import Trainer
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from pyq.experiment.classes_builder import InstancesBuilder
from pyq.io.yaml_files import YamlReader
from pyq.paths import PyqPath
from pyq.training.loop import TrainingLoop
from pyq.training.task import ImageTask


class TestDatasetParserTraining(TestCase):
    def setUp(self):
        self.builder = InstancesBuilder()

    @skip("skip downloading of the dataset")
    def test_build_dataset_from_yaml(self):
        yaml_file_path = "./test_cases/dataset_parser_training_test_case.yaml"
        class_as_dict = YamlReader(yaml_file_path).read()
        objects_as_dict = self.builder.convert_dictionary_to_instances(class_as_dict)

        dataset_initializer = objects_as_dict["dataset"]
        parser = objects_as_dict["parser"]
        model = objects_as_dict["model"]

        model = parser.apply(model)

        training_dataset, validation_dataset = dataset_initializer.get_train_test_set()
        training_loader = DataLoader(training_dataset, batch_size=16384)
        validation_loader = DataLoader(validation_dataset, batch_size=16384)

        task = ImageTask("classification", training_dataset, model)
        optimizer = optim.Adam(model.parameters(), weight_decay=5e-4, lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        losses = [
            CrossEntropyLoss(),
        ]

        training_loop = TrainingLoop(
            model=model, task=task, losses=losses, metrics=losses, optimizer=optimizer, scheduler=scheduler
        )

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        trainer = Trainer(
            default_root_dir=PyqPath.get_project_absolute_dir() + "/datasets",
            gpus=1 if str(device).startswith("cuda") else 0,
            max_epochs=1,
        )

        trainer.fit(training_loop, training_loader, validation_loader)


if __name__ == "__main__":
    main()
