import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def build_transforms(mean, std, image_size=32, augment=False, cutout=False):
    """Returns (train_transform, eval_transform).

    mean/std and image_size are supplied by the model, since a pretrained
    backbone requires the preprocessing it was pretrained with.
    """
    resize = []
    if image_size != 32:
        # Pretrained backbones expect their native resolution. Bicubic is the
        # standard choice for upscaling; nearest-neighbour would produce hard
        # blocky edges unlike anything the model saw during pretraining.
        resize = [transforms.Resize(image_size,
                                    interpolation=InterpolationMode.BICUBIC)]

    eval_tf = transforms.Compose(
        resize + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    if not augment:
        return eval_tf, eval_tf

    train_steps = [
        # Augment at 32x32 first, then upscale. Cropping and flipping small
        # images is far cheaper than doing it at 224x224, and the result is
        # equivalent.
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
    ] + resize + [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    if cutout:
        train_steps.append(
            transforms.RandomErasing(p=0.5, scale=(0.05, 0.15), value='random')
        )

    return transforms.Compose(train_steps), eval_tf


def get_loaders(config, device, mean, std, image_size):
    d, h = config['data'], config['hyperparameters']
    train_tf, eval_tf = build_transforms(mean, std, image_size,
                                         d['augment'], d['cutout'])

    # Two dataset objects over the same files carrying different transforms.
    # random_split would share a single transform between train and val,
    # silently augmenting the validation set.
    train_full = datasets.CIFAR10(root=d['root'], train=True,
                                  download=True, transform=train_tf)
    val_full = datasets.CIFAR10(root=d['root'], train=True,
                                download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(root=d['root'], train=False,
                                download=True, transform=eval_tf)

    generator = torch.Generator().manual_seed(h['seed'])
    perm = torch.randperm(len(train_full), generator=generator).tolist()
    n_val = int(len(train_full) * d['val_split'])
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)

    cuda = device.type == 'cuda'
    common = dict(
        batch_size=h['batch_size'],
        num_workers=d['num_workers'],
        pin_memory=cuda,
        persistent_workers=d['num_workers'] > 0,
    )

    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    print(f'Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}')
    print(f'Input resolution: {image_size}x{image_size}')
    return train_loader, val_loader, test_loader
