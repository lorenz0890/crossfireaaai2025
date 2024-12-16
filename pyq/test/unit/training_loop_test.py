from unittest import TestCase, main

import torch
from pytorch_lightning import Trainer
from torch import optim
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch_geometric.loader import DataLoader
from torchvision import transforms

from pyq.datasets.initializer import DataInitializer
from pyq.paths import PyqPath
from pyq.training.loop import DistilledTrainingLoop, TrainingLoop
from pyq.training.task import GraphTask, ImageTask
from pyq.utils import GCN, ConvNet


class TestTrainingLoop(TestCase):
    def setUp(self):
        pass

    def test_cora_training_loop(self):
        data_initializer = DataInitializer(dataset_name="Planetoid", name="Cora")
        training_dataset, validation_dataset = data_initializer.get_train_test_set()
        training_loader = DataLoader(training_dataset, batch_size=256)
        validation_loader = DataLoader(validation_dataset, batch_size=256)

        model = GCN(in_channels=training_dataset.num_features, out_channels=training_dataset.num_classes)
        task = GraphTask("classification", training_dataset, model)
        optimizer = optim.Adam(model.parameters(), weight_decay=5e-4, lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        losses = [
            CrossEntropyLoss(),
        ]

        training_loop = TrainingLoop(
            model=model, task=task, losses=losses, y_transform={},
            metrics=[], optimizer=optimizer, scheduler=scheduler
        )

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        trainer = Trainer(
            default_root_dir=PyqPath.get_project_absolute_dir() + "/datasets",
            gpus=1 if str(device).startswith("cuda") else 0,
            max_epochs=1,
        )

        trainer.fit(training_loop, training_loader, validation_loader)

    def test_cora_distilled_training_loop(self):
        data_initializer = DataInitializer(dataset_name="Planetoid", name="Cora")
        training_dataset, validation_dataset = data_initializer.get_train_test_set()
        training_loader = DataLoader(training_dataset, batch_size=256)
        validation_loader = DataLoader(validation_dataset, batch_size=256)

        student_model = GCN(in_channels=training_dataset.num_features, out_channels=training_dataset.num_classes)
        teacher_model = GCN(in_channels=training_dataset.num_features, out_channels=training_dataset.num_classes)
        task = GraphTask("classification", training_dataset, student_model)
        optimizer = optim.Adam(student_model.parameters(), weight_decay=5e-4, lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        training_loop = DistilledTrainingLoop(
            model=student_model,
            teacher_model=teacher_model,
            task=task,
            losses=[],
            teacher_losses=[KLDivLoss()],
            loss_coefficient=0,
            y_transform={},
            metrics=[],
            optimizer=optimizer,
            scheduler=scheduler,
        )

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        trainer = Trainer(
            default_root_dir=PyqPath.get_project_absolute_dir() + "/datasets",
            gpus=1 if str(device).startswith("cuda") else 0,
            max_epochs=1,
        )

        trainer.fit(training_loop, training_loader, validation_loader)

    def test_mnist_training_loop(self):
        data_initializer = DataInitializer(dataset_name="MNIST", transform=transforms.ToTensor())
        training_dataset, validation_dataset = data_initializer.get_train_test_set()
        training_loader = DataLoader(training_dataset, batch_size=16384)
        validation_loader = DataLoader(validation_dataset, batch_size=16384)

        model = ConvNet(1, 10)
        task = ImageTask("classification", training_dataset, model)
        optimizer = optim.Adam(model.parameters(), weight_decay=5e-4, lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        losses = [
            torch.nn.CrossEntropyLoss(),
        ]

        training_loop = TrainingLoop(
            model=model, task=task, losses=losses, y_transform={},
            metrics=[], optimizer=optimizer, scheduler=scheduler
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
