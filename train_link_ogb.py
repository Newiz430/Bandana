import argparse
from datetime import datetime
from tqdm import tqdm
from copy import copy

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset

from src.utils import Logger, set_seed, load_config, print_desc
from src.model import Bandana, Decoder, Encoder
from src.mask import BandwidthMask


def train_link(model, splits, args, device="cpu"):

    def train(data):
        model.train()
        exclude_layers = [i + 1 in args.exclude_layers if args.exclude_layers is not None else []
                          for i in range(args.encoder_layers)]
        loss = model.train_epoch(data.to(device), optimizer, neg_ratio=args.neg_ratio, temp=args.temp,
                                 exclude_layers=exclude_layers, batch_size=args.batch_size, sparse=args.sparse)
        return loss

    @torch.no_grad()
    def test(splits, batch_size=2**16):
        model.eval()
        test_data = splits['test'].to(device)
        z = model(test_data.x, test_data.edge_index)
        results = model.test_ogb(z, splits, evaluator, batch_size=batch_size)
        return results

    evaluator = Evaluator(name=args.dataset)
    checkpoint = args.checkpoint
    loggers = {
        'Hits@20': Logger('Hits@20', args.runs, now, args, log_path=args.log_path),
        'Hits@50': Logger('Hits@50', args.runs, now, args, log_path=args.log_path),
        'Hits@100': Logger('Hits@100', args.runs, now, args, log_path=args.log_path),
    }
    for run in range(args.runs):
        if not args.load_from_cp:
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)

            result_dict = {'Hits@20': None, 'Hits@50': None, 'Hits@100': None}

            bar = tqdm(range(1, 1 + args.epochs))
            for epoch in bar:

                loss = train(splits['train'])
                print_desc(bar, run, loss, result_dict)

                if epoch % args.eval_period == 0:
                    results = test(splits)
                    for key, result in results.items():
                        result_dict[key] = result
                    print_desc(bar, run, loss, result_dict)

            torch.save(model.state_dict(), checkpoint)

        model.load_state_dict(torch.load(checkpoint))
        results = test(splits)

        for key, res in results.items():
            print(f"[Test] best {key}: val = {res[0]:.2%}, test = {res[1]:.2%}")

        for key, result in results.items():
            loggers[key].add_result(run, result)

    print("\n")
    loggers['Hits@20'].print_statistics(print_info=True)
    loggers['Hits@50'].print_statistics()
    loggers['Hits@100'].print_statistics()



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="ogbl-collab", help="Datasets. (default: ogbl-collab)")
parser.add_argument("--data_path", type=str, default="./data", help="Path for dataset raw files. (default: ./data)")

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=256, help='Channels of encoder input. (default: 256)')
parser.add_argument('--decoder_channels', type=int, default=64, help='Channels of decoder intermediate layers. (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers of encoder. (default: 1)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.3, help='Dropout probability of encoder. (default: 0.3)')
parser.add_argument('--decoder_dropout', type=float, default=0.3, help='Dropout probability of decoder. (default: 0.3)')
parser.add_argument('--dense', action='store_false', dest='sparse', help='Whether to use normal tensors instead of sparse tensors for adjacent matrix. (default: False)')

parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs. (default: 1000)')
parser.add_argument('--temp', type=float, nargs="+", default=[1], help='Softmax temperature for masking. (default: 1)')
parser.add_argument('--neg_ratio', type=float, default=0.5, help='Ratio for sampling negative edges. (default: 50%)')
parser.add_argument('--exclude_layers', type=int, nargs='*', help='Encoder layers to be excluded from layer-wise loss. (default: [])')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training. (default: 0.001)')
parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay for training. (default: 0.)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='Grad norm for training. (default: 1.0)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size. (default: 2**16)')

parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')
parser.add_argument('--eval_period', type=int, default=10, help='Interval between two evaluation steps. (default: 10)')
parser.add_argument("--checkpoint", type=str, default="cp_link", help="Checkpoint save path for model. (default: cp_link)")
parser.add_argument("--load_from_cp", action='store_true', help="Only evaluate with the .pth files from `--checkpoint`. (default: False)")
parser.add_argument('--debug', action='store_true', help='Whether to log information in each epoch. (default: False)')
parser.add_argument("--device", type=int, default=0, help='GPU id. (default: 0)')
parser.add_argument("--use_cfg", action='store_true', help='Whether to use the best configurations. (default: False)')
parser.add_argument('--log_path', type=str, default='./log', help='Path for log files. (default: ./log)')

args = parser.parse_args()
if args.use_cfg:
    args = load_config(args, './config', 'link')
if not args.checkpoint.endswith('.pth'):
    args.checkpoint += '.pth'

set_seed(args.seed)
if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

transform = T.ToDevice(device)

print('Loading Data...')
if args.dataset in {'ogbl-collab', 'ogbl-ppa'}:
    dataset = PygLinkPropPredDataset(name=args.dataset, root=args.data_path)
    data = transform(dataset[0])
    del data.edge_weight, data.edge_year
else:
    raise ValueError(args.dataset)
    
split_edge = dataset.get_edge_split()
if args.dataset in {'ogbl-collab'}:
    args.year = 2010
    if args.year > 0:
        year_mask = split_edge['train']['year'] >= args.year
        split_edge['train']['edge'] = split_edge['train']['edge'][year_mask]
        data.edge_index = to_undirected(split_edge['train']['edge'].t())
        print(f"{1 - year_mask.float().mean():.2%} of edges are dropped according to edge year {args.year}.")
train_data, val_data, test_data = copy(data), copy(data), copy(data)
            
args.val_as_input = True
if args.val_as_input:
    full_edge_index = torch.cat([split_edge['train']['edge'], split_edge['valid']['edge']], dim=0).t()
    full_edge_index = to_undirected(full_edge_index)
    train_data.edge_index = full_edge_index 
    val_data.edge_index = full_edge_index
    test_data.edge_index = full_edge_index
    train_data.pos_edge_label_index = torch.cat([split_edge['train']['edge'], split_edge['valid']['edge']], dim=0).t()
else:
    train_data.pos_edge_label_index = split_edge['train']['edge'].t()

val_data.pos_edge_label_index = split_edge['valid']['edge'].t()
val_data.neg_edge_label_index = split_edge['valid']['edge_neg'].t()
test_data.pos_edge_label_index = split_edge['test']['edge'].t()
test_data.neg_edge_label_index = split_edge['test']['edge_neg'].t()

splits = dict(train=train_data, valid=val_data, test=test_data)

mask = BandwidthMask(num_nodes=data.num_nodes)
encoder = Encoder(data.num_features, args.encoder_channels,
                  num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                  bn=args.bn, activation=args.encoder_activation,
                  use_node_feats=False, node_emb=256, num_nodes=data.num_nodes)
edge_decoder = Decoder(args.encoder_channels, args.decoder_channels, out_channels=2,
                       num_layers=args.decoder_layers, dropout=args.decoder_dropout)
model = Bandana(encoder, edge_decoder, mask, random_negative_sampling=True).to(device)

now = datetime.now().strftime('%b%d_%H-%M-%S')
train_link(model, splits, args, device=device)
