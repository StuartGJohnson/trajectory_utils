import numpy as np
import sklearn.utils
import torch
from torch.utils.data import TensorDataset
from typing import Tuple

def load_dataset(fname: str) -> Tuple[TensorDataset, TensorDataset, int, int]:
    data = np.load(fname)
    x = data['first_array']
    y = data['second_array']
    z = data['third_array']

    print(x.shape, y.shape, z.shape)

    # shuffle nn data to prevent trajectory ordering
    x, y = sklearn.utils.shuffle(x, y, random_state=42)

    input_dim = x.shape[1]
    output_dim = y.shape[1] if y.ndim > 1 else 1

    xt = torch.from_numpy(x).float()
    yt = torch.from_numpy(y).float()
    zt = torch.from_numpy(z).float()

    dataset = TensorDataset(xt, yt)
    metric_dataset = TensorDataset(zt)
    return dataset, metric_dataset, input_dim, output_dim