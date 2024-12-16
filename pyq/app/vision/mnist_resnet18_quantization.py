import warnings

import torch
import torch.nn.functional as F
from loguru import logger
from torch import cat, cuda
from torch.nn import BatchNorm2d, Sequential
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor

from pyq.core.parser import TorchModelParser
from pyq.core.quantization.functional import STEOffsetQuantizeFunction, STEQuantizeFunction
from pyq.core.quantization.initializer import MinMaxInitializer, MinMaxOffsetInitializer
from pyq.core.quantization.observer import UniformRangeObserver
from pyq.core.quantization.wrapper import ActivationQuantizerWrapper, TransformationLayerQuantizerWrapper
from pyq.models.vision import ResNet18

# skip warnings
warnings.filterwarnings("ignore")


@logger.catch
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                )
            )


@logger.catch
def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


if __name__ == "__main__":
    # hyper-parameters
    bits = 4
    batch_size = 128
    fp32_epochs = 5
    int8_epochs = 5
    lr = 0.1

    # prepare default configuration for the training procedures
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": batch_size}

    if cuda.is_available():
        device = torch.device("cuda")
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        device = torch.device("cpu")

    transform = Compose([ToTensor(), Normalize([0.5], [0.1]), Lambda(lambda x: cat([x, x, x], 0))])

    train_dataset = MNIST("./datasets/", train=True, download=True, transform=transform)
    test_dataset = MNIST("./datasets/", train=False, transform=transform)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    # parse the model to obtain quantize model, and then train the model normally using your own training loop
    model = ResNet18(pretrained=True, num_classes=10).to(device)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = MultiStepLR(
        optimizer,
        milestones=[
            5,
        ],
    )

    print("_" * 35, "FP32 Training", "_" * 35)

    for epoch in range(1, fp32_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, 100)
        validate(model, device, test_loader)
        lr_scheduler.step()

    # !important: configure the quantization function and initialization
    layer_quantizer = TransformationLayerQuantizerWrapper
    layer_quantizer_arguments = {
        "quantizer_function": STEQuantizeFunction(),
        "initializer": MinMaxInitializer().to(device),
        "range_observer": UniformRangeObserver(bits=bits).to(device),
    }

    activation_quantizer = ActivationQuantizerWrapper
    activation_quantizer_arguments = {
        "quantizer_function": STEOffsetQuantizeFunction(),
        "initializer": MinMaxOffsetInitializer().to(device),
        "range_observer": UniformRangeObserver(bits=bits, is_positive=True).to(device),
    }

    skip_layer_by_type = [Sequential, BatchNorm2d, BasicBlock]
    parser = TorchModelParser(
        callable_object=layer_quantizer,
        callable_object_kwargs=layer_quantizer_arguments,
        callable_object_for_nonparametric=activation_quantizer,
        callable_object_for_nonparametric_kwargs=activation_quantizer_arguments,
        skip_layer_by_type=skip_layer_by_type,
    )

    print("_" * 35, "INT{} Training".format(bits), "_" * 35)

    model = parser.apply(model)
    optimizer = Adam(model.parameters(), lr=0.00001)

    for epoch in range(1, int8_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, 100)
        validate(model, device, test_loader)
