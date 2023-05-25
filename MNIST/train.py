import numpy as np
import jax

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from model import init_MLP, update

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
assert label.shape == (128,)

seed = 0
key = jax.random.PRNGKey(seed)
MLP_params = init_MLP([784, 512, 256, 10], key)
lr = 1e-4

epochs = 10

for epoch in range(epochs):
    for imgs, labels in train_loader:
        gt_labels = jax.nn.one_hot(labels, len(MNIST.classes))
        print(gt_labels.shape)
        loss, MLP_params = update(MLP_params, imgs, gt_labels, lr)
        print(loss)
        break
    break
