from matplotlib import pyplot as plt
import numpy as np

def visualize_dataset(ds, path, start_idx):
    nrows = 4
    ncols = 6
    size = 3

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for i in range(min(nrows * ncols, len(ds))):
        ax = axs[i // ncols][i % ncols]
        idx = start_idx + i
        batch = ds[idx]
        if len(batch) == 2:
            img, y = batch
        else:
            img, y, _ = batch
        img = np.moveaxis(ds.reverse_normalization(img).numpy(), 0, 2)
        ax.imshow(img)
        ax.set_title(y)
    fig.savefig(path)