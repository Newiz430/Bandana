import os.path as osp
import sys
import time
from datetime import datetime
import argparse

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit

from src.utils import Logger, set_seed
from src.model import Bandana, Decoder, Encoder, DotEdgeDecoder
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
        edlp = DotEdgeDecoder(args.encoder_channels, None, num_layers=1).to(device)
        edlp.eval()
        model.eval()

        train_data = splits['train'].to(device)
        z = model(train_data.x, train_data.edge_index)

        valid_auc, valid_ap = model.test(
            z, splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index, batch_size=batch_size,
            probing_decoder=edlp)

        test_auc, test_ap = model.test(
            z, splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index, batch_size=batch_size,
            probing_decoder=edlp)

        results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
        return results

    monitor = 'AUC'
    checkpoint = args.checkpoint
    loggers = {
        'AUC': Logger('AUC', args.runs, now, args, log_path=args.log_path),
        'AP': Logger('AP', args.runs, now, args, log_path=args.log_path),
    }
    print('Start Training (Link Prediction Pretext Training)...')
    for run in range(args.runs):
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
                if args.debug:
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

    print('##### Final Testing result (Link Prediction)')
    loggers['AUC'].print_statistics(print_info=True)
    loggers['AP'].print_statistics()


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument("--data_path", type=str, default="./data", help="Path for dataset raw files. (default: ./data)")
parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=64, help='Channels of encoder input. (default: 64)')
parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder intermediate layers. (default: 32)')
parser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers for encoder. (default: 1)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument('--dense', action='store_false', dest='sparse', help='Whether to use sparse tensors for adjacent matrix. (default: True)')

parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size. (default: 2**16)')

parser.add_argument('--p', type=float, nargs="+", default=[0.7], help='Mask ratios or sample ratios for MaskEdge/MaskPath')
parser.add_argument('--temp', type=float, nargs="+", default=[1], help='Softmax temperature for masking')
parser.add_argument('--neg_ratio', type=float, default=0.5, help='Ratio for sampling negative edges')
parser.add_argument('--exclude_layers', type=int, nargs='*', help='Encoder layers to be excluded from layer-wise loss. (default: [])')

parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs. (default: 1000)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=10, help='(default: 10)')
parser.add_argument('--patience', type=int, default=30, help='(default: 30)')
parser.add_argument("--checkpoint", nargs="?", default="cp_link", help="save path for model. (default: cp_link)")
parser.add_argument('--debug', action='store_true', help='Whether to log information in each epoch. (default: False)')
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--log_path', type=str, default='./log')

args = parser.parse_args()
if not args.checkpoint.endswith('.pth'):
    args.checkpoint += '.pth'

args.cmd = 'python ' + ' '.join(sys.argv)
set_seed(args.seed)
if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.ToUndirected(),
    T.ToDevice(device),
])

if args.dataset in {'arxiv'}:
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(root=args.data_path, name=f'ogbn-{args.dataset}')
elif args.dataset in {'Cora', 'Citeseer', 'Pubmed'}:
    dataset = Planetoid(args.data_path, args.dataset)
elif args.dataset == 'Reddit':
    dataset = Reddit(osp.join(args.data_path, args.dataset))
elif args.dataset in {'Photo', 'Computers'}:
    dataset = Amazon(args.data_path, args.dataset)
elif args.dataset in {'CS', 'Physics'}:
    dataset = Coauthor(args.data_path, args.dataset)
else:
    raise ValueError(args.dataset)

data = transform(dataset[0])
train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1,
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