# TODO (Samir): add the __all__ here when that controller start need some of the classes
from builtins import tuple

from pytorch_lightning.callbacks import LearningRateMonitor
from torch import float32, int32, tensor
from torch.nn import (AdaptiveAvgPool2d, BatchNorm1d, BatchNorm2d, BCEWithLogitsLoss, Conv2d, CrossEntropyLoss, Dropout,
                      Embedding, KLDivLoss, Linear, MaxPool2d, ModuleDict, ModuleList, MSELoss, NLLLoss, ReLU,
                      Sequential)
from torch.optim import SGD, Adam, RAdam, Rprop
from torch.optim.lr_scheduler import (ConstantLR, CosineAnnealingLR, CyclicLR, ExponentialLR, LambdaLR, LinearLR,
                                      MultiplicativeLR, MultiStepLR, OneCycleLR, StepLR)
from torch_geometric.nn import (GAT, GCN, GIN, MLP, Aggregation, BatchNorm, GATConv, GATv2Conv, GCNConv, GINConv,
                                GraphSAGE, JumpingKnowledge, MeanAggregation, MultiAggregation)
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.transforms import GDC, Compose, GCNNorm, NormalizeFeatures, OneHotDegree, ToDense, ToSparseTensor
from torch_geometric.typing import Adj
from torchmetrics import AUROC, ROC, Accuracy, CohenKappa, F1Score, KLDivergence, MeanSquaredError#, AUC
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import (CenterCrop, Lambda, Normalize, RandomCrop, RandomHorizontalFlip, RandomResizedCrop,
                                    RandomRotation, Resize, ToTensor)

from pyq.core import TorchModelParser, TorchModuleWrapper
from pyq.core.quantization.communication_wrapper import (CommunicationGraphQuantizerWrapper,
                                                         SamplerCommunicationGraphQuantizerWrapper)
from pyq.core.quantization.functional import (LSQPlusQuantizeFunction, LSQQuantizeFunction, PACTQuantizeFunction,
                                              STEOffsetQuantizeFunction, STEQuantizeFunction,
                                              bernoulli_probability_to_mask)
from pyq.core.quantization.initializer import (DictionaryToEmptyInitializer, FixedInitializer, FixedOffsetInitializer,
                                               LsqInitializer, LsqPlusInitializer, MinMaxInitializer,
                                               MinMaxOffsetInitializer, PACTInitializer, PACTOffsetInitializer,
                                               RandomInitializer, RandomOffsetInitializer)
from pyq.core.quantization.observer import (MinMaxUniformRangeObserver, MomentumMinMaxUniformRangeObserver,
                                            UniformRangeObserver)
from pyq.core.quantization.wrapper import (ActivationQuantizerWrapper, GenericLayerQuantizerWrapper,
                                           TransformationLayerQuantizerWrapper)
from pyq.datasets.initializer import DataInitializer
from pyq.datasets.splitter import GraphDataKFold, TorchVisionDataKFold
from pyq.datasets.transforms import MaskMaker, NormalizedDegree, ProbabilityMaskMaker, in_degree_mask
from pyq.io.model import ModelReader
from pyq.models.editor import TorchModelEditor
from pyq.models.graph import GraphAttentionNetwork, GraphConvolutionalNetwork, GraphIsomorphismNetwork
from pyq.models.graph.mlp import MultiLayerPerceptron
from pyq.models.graph.sage import SampleAndAggregateGraph
from pyq.models.vision import VGG16BN, AlexNet, MobileNetV2, ResNet18
from pyq.training.loop import DistilledTrainingLoop, TrainingLoop
from pyq.training.task import GraphTask, ImageTask
from pyq.utils import ConvNet

__all__ = [
    "AUC",
    "AUROC",
    "Accuracy",
    "ActivationQuantizerWrapper",
    "Adam",
    "AdaptiveAvgPool2d",
    "Adj",
    "Aggregation",
    "AlexNet",
    "BCEWithLogitsLoss",
    "BatchNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BasicGNN",
    "BasicBlock",
    "CenterCrop",
    "CohenKappa",
    "CommunicationGraphQuantizerWrapper",
    "Compose",
    "ConstantLR",
    "Conv2d",
    "ConvNet",
    "CosineAnnealingLR",
    "CrossEntropyLoss",
    "CyclicLR",
    "DataInitializer",
    "DictionaryToEmptyInitializer",
    "DistilledTrainingLoop",
    "DoubleIdentityFunction",
    "Dropout",
    "Embedding",
    "ExponentialLR",
    "F1Score",
    "FixedInitializer",
    "FixedOffsetInitializer",
    "GAT",
    "GATConv",
    "GATv2Conv",
    "GCN",
    "GCNConv",
    "GCNNorm",
    "GDC",
    "GIN",
    "GINConv",
    "GenericLayerQuantizerWrapper",
    "GraphAttentionNetwork",
    "GraphConvolutionalNetwork",
    "GraphDataKFold",
    "GraphIsomorphismNetwork",
    "GraphSAGE",
    "GraphTask",
    "IdentityFunction",
    "ImageTask",
    "JumpingKnowledge",
    "KLDivLoss",
    "KLDivergence",
    "LSQPlusQuantizeFunction",
    "LSQQuantizeFunction",
    "Lambda",
    "LambdaLR",
    "LearningRateMonitor",
    "Linear",
    "LinearLR",
    "LsqInitializer",
    "LsqPlusInitializer",
    "LsqPlusInitializer",
    "MLP",
    "MSELoss",
    "MultiLayerPerceptron",
    "MaskMaker",
    "MaxPool2d",
    "MeanAggregation",
    "MeanSquaredError",
    "MinMaxInitializer",
    "MinMaxOffsetInitializer",
    "MinMaxUniformRangeObserver",
    "MobileNetV2",
    "ModelReader",
    "ModuleDict",
    "ModuleList",
    "MomentumMinMaxUniformRangeObserver",
    "MultiAggregation",
    "MultiStepLR",
    "MultiplicativeLR",
    "NLLLoss",
    "Normalize",
    "NormalizeFeatures",
    "NormalizedDegree",
    "OneCycleLR",
    "OneHotDegree",
    "PACTInitializer",
    "PACTOffsetInitializer",
    "PACTQuantizeFunction",
    "ProbabilityMaskMaker",
    "RAdam",
    "ROC",
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomInitializer",
    "RandomOffsetInitializer",
    "RandomResizedCrop",
    "RandomRotation",
    "ReLU",
    "ResNet18",
    "Resize",
    "Rprop",
    "SampleAndAggregateGraph",
    "SGD",
    "STEOffsetQuantizeFunction",
    "STEQuantizeFunction",
    "SamplerCommunicationGraphQuantizerWrapper",
    "Sequential",
    "StepLR",
    "ToDense",
    "ToSparseTensor",
    "ToTensor",
    "TorchModelEditor",
    "TorchModelParser",
    "TorchModuleWrapper",
    "TorchVisionDataKFold",
    "TrainingLoop",
    "TransformationLayerQuantizerWrapper",
    "UniformRangeObserver",
    "VGG16BN",
    "bernoulli_probability_to_mask",
    "float32",
    "in_degree_mask",
    "tensor",
    "tuple",
    "int32",
]
