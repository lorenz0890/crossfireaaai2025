from torch.hub import load_state_dict_from_url
from torchvision.models import VGG as _VGG
from torchvision.models.vgg import cfgs, make_layers, model_urls


class VGG16BN(_VGG):
    def __init__(self, pretrained: bool = False, progress: bool = True, **kwargs):
        super().__init__(make_layers(cfgs["D"], batch_norm=True), **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["vgg16_bn"], progress=progress)
            # if the pretrained model has different number of classes for the output layer, so remove this layer weights
            if self.classifier[6].weight.shape[0] != state_dict["classifier.6.weight"].shape[0]:
                state_dict["classifier.6.weight"] = self.classifier[6].state_dict()["weight"]
                state_dict["classifier.6.bias"] = self.classifier[6].state_dict()["bias"]
            self.load_state_dict(state_dict)
