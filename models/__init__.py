from models.resnet import ResNet50
from models.small import LeNet, FC1024, BNNet, FC400
from models.gemresnet import GEMResNet18
from models.vgg8 import vgg8
from models.vgg import vgg16_bn
from models.alexnet import Alexnet
__all__ = [
    "LeNet",
    "FC1024",
    "BNNet",
    "ResNet50",
    "GEMResNet18",
    "VGG8",
]