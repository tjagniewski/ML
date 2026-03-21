
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import ImageFolder

def calculate_dataset_stats(data_path, batch_size=64, img_size=256):
    """
    Calculate mean and standard deviation for images in RGB format
    """

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in tqdm(loader):
        # images shape: [B, C, H, W]
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        
        eps = 0.00001

        # sum pixels' values for each channel separately
        # Dim [0, 2, 3] is respectively: Batch, Height, Width
        fst_moment = (cnt * fst_moment + torch.sum(images, dim=[0, 2, 3])) / ((cnt + nb_pixels) + eps)
        snd_moment = (cnt * snd_moment + torch.sum(images**2, dim=[0, 2, 3])) / ((cnt + nb_pixels) + eps)
        
        cnt += nb_pixels

    mean = fst_moment
    std = torch.sqrt(snd_moment - fst_moment**2)

    return mean, std

def create_data_loaders(data_dir, batch_size=32, num_workers=4): # Domyślnie zmienione na 4
    data_transform = transforms.Compose(
        [ 
        transforms.Resize((256,256)), # ZMIANA KOLEJNOŚCI: Najpierw Resize...
        transforms.ToTensor(),        # ...potem ToTensor
        transforms.Normalize((0.4680, 0.4680, 0.4680), (0.2473, 0.2473, 0.2473))
        ]
    )

    trainset = ImageFolder(data_dir + "train/aug", transform=data_transform)
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False # NOWOŚĆ
    )

    valset = ImageFolder(data_dir + "val", transform=data_transform)
    val_loader = DataLoader(
        valset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, # POPRAWKA: Brakowało num_workers
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    testset = ImageFolder(data_dir + "test", transform=data_transform)
    test_loader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, # POPRAWKA: Brakowało num_workers
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, test_loader