import torchvision.transforms as T

# Standard ImageNet normalization statistics used by pre-trained ResNet-50
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Creates a composition of data augmentation transforms for training.

    Transforms included:
    1. Resize: Resizes image to slightly larger than target size for cropping.
    2. Random Crop: Crops a random patch of target size to make model translation-invariant.
    3. Random Horizontal Flip: Flips image horizontally (mimics left/right eye symmetry).
    4. Random Rotation: Rotates image to handle head tilts/camera alignment.
    5. Color Jitter: Randomly alters brightness and contrast to mimic varying lighting.
    6. ToTensor: Converts PIL Image to PyTorch Tensor.
    7. Normalize: Standardizes tensor values to match ImageNet distribution.
    """
    return T.Compose(
        [
            T.Resize((int(image_size * 1.15), int(image_size * 1.15))),
            T.RandomCrop((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=(-15, 15)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """
    Creates a composition of preprocessing transforms for validation and testing.
    Note: NO random augmentations are applied here to keep evaluation deterministic.
    """
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
