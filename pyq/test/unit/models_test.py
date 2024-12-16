from unittest import TestCase, main

import torch

from pyq.models.graph import GraphAttentionNetwork, GraphConvolutionalNetwork, GraphIsomorphismNetwork


class TestGraphModels(TestCase):
    def setUp(self):
        self.loss = torch.nn.CrossEntropyLoss()
        self.training_dataset_num_features = 10
        self.training_dataset_num_classes = 5

        x, edge_index = torch.rand(20, 10), torch.randint(0, 1, (2, 100))
        batch = torch.randint(0, 1, (20, self.training_dataset_num_classes))
        self.data_point = (x, edge_index, batch)

    def test_gcn_model_forward(self):
        model = GraphConvolutionalNetwork(
            in_channels=self.training_dataset_num_features,
            hidden_channels=self.training_dataset_num_classes,
            out_channels=self.training_dataset_num_classes,
            module_num_layers=2,
        )
        y = model(*self.data_point)
        self.assertTrue(y is not None)

    def test_gcn_model_backward(self):
        model = GraphConvolutionalNetwork(
            in_channels=self.training_dataset_num_features,
            hidden_channels=self.training_dataset_num_classes,
            out_channels=self.training_dataset_num_classes,
            module_num_layers=2,
        )
        y = model(*self.data_point)
        loss = self.loss(y, torch.rand_like(y))
        loss.backward()
        params = [*model.parameters()]
        self.assertTrue(all(param.grad is not None for param in params))

    def test_gin_model_forward(self):
        model = GraphIsomorphismNetwork(
            in_channels=self.training_dataset_num_features,
            hidden_channels=self.training_dataset_num_classes,
            out_channels=self.training_dataset_num_classes,
            num_layers=16,
        )
        y = model(*self.data_point)
        self.assertTrue(y is not None)

    def test_gin_model_backward(self):
        model = GraphIsomorphismNetwork(
            in_channels=self.training_dataset_num_features,
            hidden_channels=self.training_dataset_num_classes,
            out_channels=self.training_dataset_num_classes,
            num_layers=3,
        )
        y = model(*self.data_point)
        loss = self.loss(y, torch.rand_like(y))
        loss.backward()
        params = [*model.parameters()]
        self.assertTrue(all(param.grad is not None for param in params))

    def test_gat_model_forward(self):
        model = GraphAttentionNetwork(
            in_channels=self.training_dataset_num_features,
            hidden_channels=self.training_dataset_num_classes,
            out_channels=self.training_dataset_num_classes,
            num_layers=16,
        )
        y = model(*self.data_point)
        self.assertTrue(y is not None)

    def test_gat_model_backward(self):
        model = GraphAttentionNetwork(
            in_channels=self.training_dataset_num_features,
            hidden_channels=self.training_dataset_num_classes,
            out_channels=self.training_dataset_num_classes,
            num_layers=16,
        )
        y = model(*self.data_point)
        loss = self.loss(y, torch.rand_like(y))
        loss.backward()
        params = [*model.parameters()]
        self.assertTrue(all(param.grad is not None for param in params))


if __name__ == "__main__":
    main()
