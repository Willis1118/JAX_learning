import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

def to_flatten_nparr(x):
    '''
        transform PILImage to nparray
    '''
    return np.ravel(np.array(x, dtype=np.float32))

def collate_fn(data):
    print(type(data))
    print(data[0])
    # print(data[1].shape)

batch_size = 128
train_dataset = MNIST(root='train_mnist', train=True, download=True, transform=to_flatten_nparr)
test_dataset = MNIST(root='test_mnist', train=False, download=True, transform=to_flatten_nparr)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

batch_data = next(iter(train_loader))