import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit

from src.utils import Logger, set_seed, load_config, print_desc
from src.model import Bandana, Decoder, Encoder, DotEdgeDecoder
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
    for run in range(args.runs):
        if not args.load_from_cp:
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)

            best_valid = 0.0
            best_epoch = 0
            cnt_wait = 0
            result_dict = {'AUC': None, 'AP': None}

            bar = tqdm(range(1, 1 + args.epochs))
            for epoch in bar:

                loss = train(splits['train'])
                print_desc(bar, run, loss, result_dict, monitor, best_valid, best_epoch)

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
                        result_dict[key] = result
                    print_desc(bar, run, loss, result_dict, monitor, best_valid, best_epoch)
                    if cnt_wait == args.patience:
                        bar.close()
                        print(f'Training ends by early stopping at ep {epoch}.')
                        break

        model.load_state_dict(torch.load(checkpoint))
        results = test(splits)

        for key, res in results.items():
            print(f"[Test] best {key}: val = {res[0]:.2%}, test = {res[1]:.2%}")

        for key, result in results.items():
            loggers[key].add_result(run, result)

    loggers['AUC'].print_statistics(print_info=True)
    loggers['AP'].print_statistics()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
    parser.add_argument("--data_path", type=str, default="./data", help="Path for dataset raw files. (default: ./data)")

    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
    parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
    parser.add_argument('--encoder_channels', type=int, default=64, help='Channels of encoder input. (default: 64)')
    parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder intermediate layers. (default: 32)')
    parser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers for encoder. (default: 1)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument('--dense', action='store_false', dest='sparse', help='Whether to use normal tensors instead of sparse tensors for adjacent matrix. (default: False)')

    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs. (default: 1000)')
    parser.add_argument('--temp', type=float, nargs="+", default=[1], help='Softmax temperature for masking. (default: 1)')
    parser.add_argument('--neg_ratio', type=float, default=0.5, help='Ratio for sampling negative edges. (default: 50%)')
    parser.add_argument('--exclude_layers', type=int, nargs='*', help='Encoder layers to be excluded from layer-wise loss. (default: [])')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay for training. (default: 5e-5)')
    parser.add_argument('--grad_norm', type=float, default=1.0, help='Grad norm for training. (default: 1.0)')
    parser.add_argument('--batch_size', type=int, default=2 ** 16, help='Number of batch size. (default: 2**16)')

    parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')
    parser.add_argument('--eval_period', type=int, default=10, help='Interval between two evaluation steps. (default: 10)')
    parser.add_argument('--patience', type=int, default=30, help='Patience epochs of early stopping. (default: 30)')
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

    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ])

    if args.dataset in {'arxiv'}:
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(root=args.data_path, name=f'ogbn-{args.dataset}')
    elif args.dataset in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(args.data_path, args.dataset)
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
                      bn=args.bn, activation=args.encoder_activation)
    decoder = Decoder(args.encoder_channels, args.decoder_channels, out_channels=2,
                      num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    model = Bandana(encoder, decoder, mask=mask).to(device)

    now = datetime.now().strftime('%b%d_%H-%M-%S')
    train_link(model, splits, args, device=device)