import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

def to_flatten_nparr(x):
    '''
        transform PILImage to nparray
    '''
    return np.ravel(np.array(x, dtype=np.float32))

def collate_fn(data):
    '''
        input: data -> [(image, label)] * batch_size
    '''

    # will turn [(image, lable)] --> ([images], [labels])
    # zip ((,),(,),(,)...) will take the first and second element in every tuple and zip into a tuple of two lists
    transposed_data = list(zip(*data))

    print(transposed_data)

    images, labels = np.stack(transposed_data[0]), np.array(transposed_data[1])

    return images, labels

batch_size = 128
train_dataset = MNIST(root='train_mnist', train=True, download=True, transform=to_flatten_nparr)
test_dataset = MNIST(root='test_mnist', train=False, download=True, transform=to_flatten_nparr)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

img, label = next(iter(train_loader))

assert img.shape == (128, 784)
assert label.shape == (128, 1)