from torchvision.datasets import MNIST

train_dataset = MNIST(root='train_mnist', train=True, download=True, transform=None)
print(type(train_dataset))