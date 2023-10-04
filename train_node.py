import os.path as osp
import sys
import time
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.utils import Logger, set_seed, load_config
from src.model import Bandana, Decoder, Encoder
from src.mask import BandwidthMask


def train_linkpred(model, splits, args, device="cpu"):

    def train(data):
        model.train()
        exclude_layers = [i + 1 in args.exclude_layers if args.exclude_layers is not None else []
                          for i in range(args.encoder_layers)]
        loss = model.train_epoch(data.to(device), optimizer, neg_ratio=args.neg_ratio, mask_ratio=args.p, t=args.temp,
                                 exclude_layers=exclude_layers, batch_size=args.batch_size,
                                 sparse=args.sparse)
        return loss

    @torch.no_grad()
    def test(splits, batch_size=2**16):
        model.eval()
        train_data = splits['train'].to(device)
        z = model(train_data.x, train_data.edge_index)

        valid_auc, valid_ap = model.test(
            z, splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index, batch_size=batch_size)

        test_auc, test_ap = model.test(
            z, splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index, batch_size=batch_size)

        results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
        return results

    monitor = 'AUC'
    checkpoint = args.checkpoint
    runs = 1
    loggers = {
        'AUC': Logger('AUC', runs, now, args),
        'AP': Logger('AP', runs, now, args),
    }
    print('Start Training (Link Prediction Pretext Training)...')
    for run in range(runs):
        if not args.load_from_cp:
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)

            best_valid = 0.0
            best_epoch = 0
            cnt_wait = 0

            for epoch in range(1, 1 + args.epochs):

                t1 = time.time()
                loss = train(splits['train'])
                t2 = time.time()

                if epoch % args.eval_period == 0:
                    results = test(splits)
                    valid_result = results[monitor][0]
                    if valid_result >= best_valid:
                        best_valid = valid_result
                        best_epoch = epoch
                        torch.save(model.state_dict(), checkpoint)
                        cnt_wait = 0
                    else:
                        cnt_wait += 1

                    for key, result in results.items():
                        valid_result, test_result = result
                        print(key)
                        print(f'Run: {run + 1:02d} / {args.runs:02d}, '
                              f'Epoch: {epoch:02d} / {args.epochs:02d}, '
                              f'Best_epoch: {best_epoch:02d}, '
                              f'Best_valid: {best_valid:.2%}%, '
                              f'Loss: {loss:.4f}, '
                              f'Valid: {valid_result:.2%}, '
                              f'Test: {test_result:.2%}',
                              f'Training Time/epoch: {t2-t1:.3f}')
                    print('#' * round(140*epoch/(args.epochs+1)))
                    if cnt_wait == args.patience:
                        print('Early stopping!')
                        break

        print('##### Testing on {}/{}'.format(run + 1, args.runs))

        model.load_state_dict(torch.load(checkpoint))
        results = test(splits)

        for key, result in results.items():
            valid_result, test_result = result
            print(key)
            print(f'**** Testing on Run: {run + 1:02d}, '
                  f'Best Epoch: {best_epoch:02d}, '
                  f'Valid: {valid_result:.2%}, '
                  f'Test: {test_result:.2%}')

        for key, result in results.items():
            loggers[key].add_result(run, result)

    print('##### Final Testing result (Link Prediction Pretext Training)')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


def train_nodeclas(model, data, args, device='cpu'):
    def train(loader):
        clf.train()
        for nodes in loader:
            optimizer.zero_grad()
            loss_fn(clf(embedding[nodes]), y[nodes]).backward()
            optimizer.step()

    @torch.no_grad()
    def test(loader):
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            logits.append(clf(embedding[nodes]))
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)

        def acc(y_true, y_pred):
            return (y_pred == y_true).float().mean().item()

        def micro_f1(y_true, y_pred):
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)
            micro_f1 = f1_score(y_true=y_true.cpu(), y_pred=y_pred.cpu(), average='micro')
            return micro_f1

        def macro_f1(y_true, y_pred):
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)
            macro_f1 = f1_score(y_true=y_true.cpu(), y_pred=y_pred.cpu(), average='macro')
            return macro_f1

        return acc(labels, logits), micro_f1(labels, logits), macro_f1(labels, logits)

    # 设置数据加载器
    if hasattr(data, 'train_mask'):
        train_loader = DataLoader(data.train_mask.nonzero().squeeze(), pin_memory=False, batch_size=512, shuffle=True)
        test_loader = DataLoader(data.test_mask.nonzero().squeeze(), pin_memory=False, batch_size=20000, shuffle=False)
        val_loader = DataLoader(data.val_mask.nonzero().squeeze(), pin_memory=False, batch_size=20000, shuffle=False)
    else:
        train_loader = DataLoader(data.train_nodes.squeeze(), pin_memory=False, batch_size=4096, shuffle=True)
        test_loader = DataLoader(data.test_nodes.squeeze(), pin_memory=False, batch_size=20000, shuffle=False)
        val_loader = DataLoader(data.val_nodes.squeeze(), pin_memory=False, batch_size=20000, shuffle=False)

    data = data.to(device)
    y = data.y.squeeze()
    embedding = model.encoder.get_embedding(data.x, data.edge_index, l2_norm=args.l2_norm)

    loss_fn = nn.CrossEntropyLoss()
    clf = nn.Linear(embedding.size(1), y.max().item() + 1).to(device)

    loggers = {
        'ACC': Logger('ACC', args.runs, now, args, log_path=args.log_path),
        'MICRO-F1': Logger('MICRO-F1', args.runs, now, args, log_path=args.log_path),
        'MACRO-F1': Logger('MACRO-F1', args.runs, now, args, log_path=args.log_path),
    }
    # logger = Logger(args.runs, now, args, log_path=args.log_path)

    print('Start Training (Node Classification)...')
    for run in range(args.runs):
        nn.init.xavier_uniform_(clf.weight.data)
        nn.init.zeros_(clf.bias.data)
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=args.weight_decay_prob)  # 1 for citeseer

        best_val_metrics = [0, 0, 0]
        best_test_metrics = [0, 0, 0]
        start = time.time()
        for epoch in range(1, 101):
            train(train_loader)
            val_metrics = test(val_loader)
            test_metrics = test(test_loader)
            for i in range(3):
                if val_metrics[i] >= best_val_metrics[i]:
                    best_val_metrics[i] = val_metrics[i]
                    best_test_metrics[i] = test_metrics[i]
            end = time.time()
            if args.debug:
                print(f"Epoch {epoch:02d} / {100:02d}, Time elapsed {end-start:.4f}\n"
                      f"(ACC) Valid: {val_metrics[0]:.2%}, Test {test_metrics[0]:.2%}, Best {best_test_metrics[0]:.2%}\n"
                      f"(MICRO-F1) Valid: {val_metrics[1]:.2%}, Test {test_metrics[1]:.2%}, Best {best_test_metrics[1]:.2%}\n"
                      f"(MACRO-F1) Valid: {val_metrics[2]:.2%}, Test {test_metrics[2]:.2%}, Best {best_test_metrics[2]:.2%}"
                      )

        print(f"Run {run+1}: Best test acc {best_test_metrics[0]:.2%}, best test micro-f1 {best_test_metrics[1]:.2%}, "
              f"Best test macro-f1 {best_test_metrics[2]:.2%}.")

        loggers['ACC'].add_result(run, (best_val_metrics[0], best_test_metrics[0]))
        loggers['MICRO-F1'].add_result(run, (best_val_metrics[1], best_test_metrics[1]))
        loggers['MACRO-F1'].add_result(run, (best_val_metrics[2], best_test_metrics[2]))

    print('##### Final Testing result (Node Classification)')
    print('')
    loggers['ACC'].print_statistics(print_info=True)
    loggers['MICRO-F1'].print_statistics()
    loggers['MACRO-F1'].print_statistics()


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument("--data_path", type=str, default="./data", help="Path for dataset raw files. (default: ./data)")
parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=64, help='Channels of encoder input. (default: 64)')
parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder intermediate layers. (default: 32)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoder. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument('--dense', action='store_false', dest='sparse', help='Whether to use sparse tensors for adjacent matrix. (default: True)')

parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size for link prediction training. (default: 2**16)')

parser.add_argument('--p', type=float, nargs="+", default=[0.7], help='Mask ratios or sample ratios for MaskEdge/MaskPath')
parser.add_argument('--temp', type=float, nargs="+", default=[1], help='Softmax temperature for masking')
parser.add_argument('--neg_ratio', type=float, default=0.5, help='Ratio for sampling negative edges')
parser.add_argument('--exclude_layers', type=int, nargs='*', help='Encoder layers to be excluded from layer-wise loss. (default: [])')

parser.add_argument('--l2_norm', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
parser.add_argument('--weight_decay_prob', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs. (default: 1000)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=30, help='(default: 30)')
parser.add_argument('--patience', type=int, default=30, help='(default: 30)')
parser.add_argument("--checkpoint", nargs="?", default="cp_node", help="checkpoint save path for model. (default: cp_node)")
parser.add_argument("--load_from_cp", action='store_true', help="Only evaluate with the .pth files from `--checkpoint`. (default: False)")
parser.add_argument('--debug', action='store_true', help='Whether to log information in each epoch. (default: False)')
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--use_cfg", action='store_true', help='Whether to use the best configurations. (default: False)')
parser.add_argument('--log_path', type=str, default='./log')

args = parser.parse_args()
if args.use_cfg:
    args = load_config(args, './config', 'node')
if not args.checkpoint.endswith('.pth'):
    args.checkpoint += '.pth'

set_seed(args.seed)
if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.ToUndirected(),
    T.ToDevice(device),
])

root = osp.join(osp.expanduser('~/public_data/pyg_data'))

if args.dataset in {'ogbn-arxiv'}:
    from ogb.nodeproppred import PygNodePropPredDataset
    print('Loading ogb dataset...')
    dataset = PygNodePropPredDataset(root=args.data_path, name=args.dataset)
    data = transform(dataset[0])
    split_idx = dataset.get_idx_split()
    data.train_nodes = split_idx['train']
    data.val_nodes = split_idx['valid']
    data.test_nodes = split_idx['test']
elif args.dataset in {'Cora', 'Citeseer', 'Pubmed'}:
    dataset = Planetoid(args.data_path, args.dataset)
    data = transform(dataset[0])
elif args.dataset in {'Photo', 'Computers'}:
    dataset = Amazon(args.data_path, args.dataset)
    data = transform(dataset[0])
    data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
elif args.dataset in {'CS', 'Physics'}:
    dataset = Coauthor(args.data_path, args.dataset)
    data = transform(dataset[0])
    data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
else:
    raise ValueError(args.dataset)

train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=True)(data)

splits = dict(train=train_data, valid=val_data, test=test_data)

mask = BandwidthMask(num_nodes=data.num_nodes, undirected=True)
encoder = Encoder(data.num_features, args.encoder_channels,
                  num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                  bn=args.bn, layer=args.layer, activation=args.encoder_activation)

decoder = Decoder(args.encoder_channels, args.decoder_channels, out_channels=2,
                  num_layers=args.decoder_layers, dropout=args.decoder_dropout)

model = Bandana(encoder, decoder, mask=mask).to(device)

now = datetime.now().strftime('%b%d_%H-%M-%S')
train_linkpred(model, splits, args, device=device)
train_nodeclas(model, data, args, device=device)