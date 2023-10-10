import sys
import os
import yaml
import random
import numpy as np
import torch
from typing import Optional

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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_config(args, path, mode):

    config_file = os.path.join(path, f'{args.dataset}.yml')
    with open(config_file, "r") as f:
        conf = yaml.load(f, yaml.FullLoader)
    conf = conf[mode]
    
    float_set = {
        "decoder_dropout", "encoder_dropout", "lr", "temp", "weight_decay", "weight_decay_prob"
    }
    for k, v in conf.items():
        if k in float_set:
            v = float(v)
        setattr(args, k, v)

    return args


def print_desc(bar, run, loss, result_dict, monitor=None, best_valid=None, best_epoch=None, prefix="Pre-training"):
    desc = f"[{prefix} #{run + 1:02d}] loss = {loss:.4f}, {'/'.join(result_dict.keys())}: "
    val, test = [], []
    for key, res in result_dict.items():
        val.append(f"{res[0]:.2%}" if res is not None else "NA")
        test.append(f"{res[1]:.2%}" if res is not None else "NA")
    desc += f"val = {'/'.join(val)}, test = {'/'.join(test)}"
    if best_valid is not None:
        desc += f", best valid {monitor} = {best_valid:.2%} at ep {best_epoch:02d}"
    bar.set_description(desc)


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

    def print_statistics(self, f=sys.stdout, last_best=False, print_info=False, percentage=True):
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
        print(f' Final {self.name}: val = {best_result[:, 0].mean():.2f} ± {best_result[:, 0].std():.2f}, '
              f'test = {best_result[:, 1].mean():.2f} ± {best_result[:, 1].std():.2f}', file=f)

        if self.log_path is not None:
            with open(os.path.join(self.log_path, 'log.txt'), 'a') as log:
                log.write(f'[{self.time}]{self.info}:\n' if print_info else '')
                log.write(f'({self.name}): val = {best_result[:, 0].mean():.2f} ± {best_result[:, 0].std():.2f}\t'
                          f'test = {best_result[:, 1].mean():.2f} ± {best_result[:, 1].std():.2f}\n')