import random
import numpy as np
import os
import torch
# from matplotlib import pyplot as plt


def transdata(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    elif isinstance(data, list):
        data = np.array(data)

    return data.reshape(-1)


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def show_flow(data, title=None, name=None):
    if name:
        plt.title(name)
    if isinstance(data, tuple):
        for d in data:
            temp = transdata(d)
            plt.plot(list(range(len(temp))), temp)
    else:
        data = transdata(data)
        plt.plot(list(range(len(data))), data)
    plt.title(title)
    plt.show()


def show_density(data, bins=20, name=None):
    if name:
        plt.title(name)
    if isinstance(data, tuple):
        for d in data:
            plt.hist(transdata(d), bins=bins, alpha=0.7)
    else:
        plt.hist(transdata(data), bins=bins)
    plt.show()
