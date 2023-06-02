import jax

'''
    transform pytorch batches from tensor to numpy array
    shape: [N,C,H,W] -> [N,H,W,C]
'''

def parse_batch(batch):
    images, labels = batch
    images = images.permute([0, 2, 3, 1])  # nchw -> nhwc
    batch = {'images': images, 'labels': labels}
    # to (local_devices, device_batch_size, height, width, 3)
    batch = prepare_pt_data(batch)
    return batch

def prepare_pt_data(xs):
    """Convert a input batch from PyTorch Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x.numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)