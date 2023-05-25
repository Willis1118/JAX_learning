import numpy as np

from torchvision.datasets import MNIST

def to_nparr(x):
    '''
        transform PILImage to nparray
    '''
    return np.array(x, dtype=np.float32)

train_dataset = MNIST(root='train_mnist', train=True, download=True, transform=to_nparr)
test_dataset = MNIST(root='test_mnist', train=False, download=True, transform=to_nparr)

