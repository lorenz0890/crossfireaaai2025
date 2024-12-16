from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet as _ResNet
from torchvision.models.resnet import BasicBlock, model_urls


class ResNet18(_ResNet):
    def __init__(self, pretrained: bool = False, progress: bool = True, **kwargs):
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["resnet18"], progress=progress)
            # if the pretrained model has different number of classes for the output layer, so remove this layer weights
            if self.fc.weight.shape[0] != state_dict["fc.weight"].shape[0]:
                state_dict["fc.weight"] = self.fc.state_dict()["weight"]
                state_dict["fc.bias"] = self.fc.state_dict()["bias"]
            self.load_state_dict(state_dict)
