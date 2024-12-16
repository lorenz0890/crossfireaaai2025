from torch.hub import load_state_dict_from_url
from torchvision.models import MobileNetV2 as _MobileNetV2
from torchvision.models.mobilenetv2 import model_urls


class MobileNetV2(_MobileNetV2):
    def __init__(self, pretrained: bool = False, progress: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["mobilenet_v2"], progress=progress)
            # if the pretrained model has different number of classes for the output layer, so remove this layer weights
            if self.classifier[1].weight.shape[0] != state_dict["classifier.1.weight"].shape[0]:
                state_dict["classifier.1.weight"] = self.classifier[1].state_dict()["weight"]
                state_dict["classifier.1.bias"] = self.classifier[1].state_dict()["bias"]
            self.load_state_dict(state_dict)
