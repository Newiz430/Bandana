import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import to_undirected

from src.utils import edgeidx2sparse, softmax_with_temp


class BandwidthMask(nn.Module):
    def __init__(self, num_nodes: int = None, undirected: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.undirected = undirected

    def forward(self, edge_index, mask_ratio, t, **kwargs):
        # treat every undirected graph as a directed graph
        # (two inversed edges have different bandwidths)
        if self.undirected:
            edge_index = to_undirected(edge_index)
        veil = self._veil_edge(edge_index, t=t)

        return veil

    def _veil_edge(self, edge_index: Tensor, t: float = 1., **kwargs):
        e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        bandwidth = torch.randn_like(e_ids, dtype=torch.float32)
        edge_index = edgeidx2sparse(edge_index, self.num_nodes)
        bandwidth = softmax_with_temp(bandwidth,
                                      edge_index.storage.row(),
                                      edge_index.storage.rowptr(),
                                      edge_index.size(0),
                                      t=t)
        return bandwidth

    def extra_repr(self):
        return f"undirected={self.undirected}"
