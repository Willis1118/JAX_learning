import numpy as np
import jax

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from model import init_MLP, update, accuracy

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
    drop_last=True # --> drop the last 
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=True
)

img, label = next(iter(train_loader))

assert img.shape == (128, 784)
assert label.shape == (128,)

seed = 42
key = jax.random.PRNGKey(seed)
MLP_params = init_MLP([784, 512, 256, 10], key)
lr = 1e-2

epochs = 10
train_steps = 0
log_every = 100
test_every = 500

avg_loss = 0
avg_acc = 0

for epoch in range(epochs):
    for imgs, labels in train_loader:
        gt_labels = jax.nn.one_hot(labels, len(MNIST.classes))
        loss, MLP_params = update(MLP_params, imgs, gt_labels, lr)

        avg_loss += loss

        if train_steps % log_every == 0 and train_steps > 0:
            print(f"Epoch: {epoch}, Steps: {train_steps}, Loss: {loss}")
        
        if train_steps % test_every == 0 and train_steps > 0:
            acc = accuracy(MLP_params, test_loader)
            avg_acc += acc
            print(f"Epoch: {epoch}, Steps: {train_steps}, Acc: {acc}")
        
        train_steps += 1
    
    print(f"Epoch: {epoch}, Avg Loss: {avg_loss / len(train_loader)}, Avg Acc: {avg_acc / len(train_loader) * test_every}")

