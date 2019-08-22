import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

max_nodes = 100


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ENZYMES_d')
dataset = TUDataset(
    path,
    name='ENZYMES',
    transform=T.ToDense(max_nodes),
    pre_filter=MyFilter())
dataset = dataset.shuffle()
print(dataset)
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=20)
val_loader = DenseDataLoader(val_dataset, batch_size=20)
train_loader = DenseDataLoader(train_dataset, batch_size=20)
for data in train_loader:
    print(data)

