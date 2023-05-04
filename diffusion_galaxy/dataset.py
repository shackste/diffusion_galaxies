from torchvision import datasets, transforms
import torch


def create_pytorch_dataset(path_to_images, image_size):
    """Create a PyTorch dataset from a directory of images. The images are cropped equally on each side to be square, corresponding to the specified image size. """
    transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize(mean=[0.5], std=[0.5])

    ])
    dataset = datasets.ImageFolder(path_to_images, transform)
    return dataset


def create_pytorch_dataloader(path_to_images, image_size, batch_size=64, shuffle=True, num_workers=4):
    """Create a PyTorch dataloader from a directory of images. The images are cropped equally on each side to be square, corresponding to the specified image size."""

    dataset = create_pytorch_dataset(path_to_images, image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
