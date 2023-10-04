import sys
import os
import torch
import random
import yaml
import numpy as np
from typing import Optional
from texttable import Texttable

from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes


def edgeidx2sparse(edge_index, num_nodes, edge_attr=None):
     sparse_edge_index = SparseTensor.from_edge_index(
         edge_index, sparse_sizes=(num_nodes, num_nodes)
     ).to(edge_index.device)
     sparse_edge_index.set_value_(edge_attr, layout="coo")
     return sparse_edge_index

def softmax_with_temp(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
    t: float = 1.
) -> Tensor:
    assert t > 0
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.view(size)
        src_max = segment(src.detach(), ptr, reduce='max')
        src_max = src_max.repeat_interleave(count, dim=dim)
        out = ((src - src_max) / t).exp()
        out_sum = segment(out, ptr, reduce='sum') + 1e-16
        out_sum = out_sum.repeat_interleave(count, dim=dim)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
        out = src - src_max.index_select(dim, index)
        out = (out / t).exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / out_sum

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def load_config(args, path, mode):

    config_file = os.path.join(path, args.dataset)
    with open(config_file, "r") as f:
        conf = yaml.load(f, yaml.FullLoader)
    conf = conf[mode]

    for k, v in conf.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)

    return args

class Logger(object):
    def __init__(self, name, runs, time, info=None, log_path=None):
        self.name = name
        self.info = info
        self.results = [[] for _ in range(runs)]
        if log_path is not None and not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_path = log_path
        self.time = time

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, f=sys.stdout, last_best=False, print_info=False, percentage=True):
        if run is not None:
            result = (100 if percentage else 1) * torch.tensor(self.results[run])
            if last_best:
                # get last max value index by reversing result tensor
                argmax = result.size(0) - result[:, 0].flip(dims=[0]).argmax().item() - 1
            else:
                argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:', file=f)
            print(f'{self.name} Highest Valid: {result[:, 0].max():.2f}', file=f)
            print(f'{self.name} Highest Eval Point: {argmax + 1}', file=f)
            print(f'{self.name}    Final Test: {result[argmax, 1]:.2f}', file=f)

        else:
            result = (100 if percentage else 1) * torch.tensor(self.results)

            best_results = []

            for r in result:
                valid = r[:, 0].max().item()
                if last_best:
                    # get last max value index by reversing result tensor
                    argmax = r.size(0) - r[:, 0].flip(dims=[0]).argmax().item() - 1
                else:
                    argmax = r[:, 0].argmax().item()
                test = r[argmax, 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)
            print(f'All runs:', file=f)
            r = best_result[:, 0]
            print(f'{self.name} Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
            r = best_result[:, 1]
            print(f'{self.name}    Final Test: {r.mean():.2f} ± {r.std():.2f}', file=f)

            if self.log_path is not None:
                with open(os.path.join(self.log_path, 'log.txt'), 'a') as log:
                    log.write(f'[{self.time}]{self.info}:\n' if print_info else '')
                    log.write(f'({self.name})\tval = {best_result[:, 0].mean():.2f}±{best_result[:, 0].std():.2f}\t')
                    log.write(f'test = {best_result[:, 1].mean():.2f}±{best_result[:, 1].std():.2f}\n')