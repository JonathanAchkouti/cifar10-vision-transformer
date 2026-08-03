import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SimpleNet(nn.Module):
    """The original baseline: the classic PyTorch-tutorial LeNet-style CNN."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def conv_block(in_ch, out_ch, pool=False):
    """3x3 conv -> batch norm -> ReLU, optionally followed by 2x2 max pool."""
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    """Two conv blocks whose output is added back to their input."""

    def __init__(self, channels):
        super().__init__()
        self.block1 = conv_block(channels, channels)
        self.block2 = conv_block(channels, channels)

    def forward(self, x):
        return x + self.block2(self.block1(x))


class ResNet9(nn.Module):
    """A residual CNN sized for 32x32 inputs. Roughly 6.6M parameters."""

    def __init__(self, num_classes=10, scale=0.125):
        super().__init__()
        self.stem = conv_block(3, 64)
        self.layer1 = conv_block(64, 128, pool=True)
        self.res1 = Residual(128)
        self.layer2 = conv_block(128, 256, pool=True)
        self.layer3 = conv_block(256, 512, pool=True)
        self.res2 = Residual(512)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.scale = scale

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(self.layer1(x))
        x = self.layer2(x)
        x = self.res2(self.layer3(x))
        x = self.pool(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        return self.fc(x) * self.scale


MODELS = {
    'simplenet': SimpleNet,
    'resnet9': ResNet9,
}

# Statistics used by our from-scratch models, measured on CIFAR-10 itself.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_model(config, num_classes=10):
    """Builds either a from-scratch model or a pretrained backbone.

    Returns (model, mean, std, image_size). The normalization statistics and
    input resolution are returned alongside the model because a pretrained
    network must receive data preprocessed exactly the way it was during
    pretraining -- feeding CIFAR statistics to an ImageNet-pretrained model
    shifts every input away from the distribution its features expect.
    """
    m = config['model']
    name = m['name']

    if not m.get('pretrained', False):
        if name not in MODELS:
            raise ValueError(f"Unknown model '{name}'. Options: {list(MODELS)}")
        return MODELS[name](num_classes=num_classes), CIFAR10_MEAN, CIFAR10_STD, 32

    import timm
    model = timm.create_model(name, pretrained=True, num_classes=num_classes)
    # Ask timm what preprocessing this specific checkpoint was trained with,
    # rather than assuming. Different pretrained weights use different
    # statistics -- some ImageNet, some simply 0.5 for every channel.
    cfg = timm.data.resolve_data_config({}, model=model)
    return model, cfg['mean'], cfg['std'], cfg['input_size'][1]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
