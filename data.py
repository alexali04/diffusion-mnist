from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass
from argparse import Namespace

@dataclass
class DataConfig:
    BATCH_SIZE: int = 128

def construct_config(n: Namespace):
    return DataConfig(
      BATCH_SIZE=n.bsz
    )

def load_data(data_config):
    tfm = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5]),
    ])

    train_ds = datasets.MNIST(root='./data', train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=data_config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=data_config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_ds, test_ds