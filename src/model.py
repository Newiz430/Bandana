from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, negative_sampling
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from .base_model import GCNConv
from .utils import edgeidx2sparse


def ce_loss(pos_out, neg_out, pos_label=None):
    if pos_label is not None:
        if len(pos_out.size()) == 2 and len(neg_out.size()) == 2:
            assert pos_out.size(1) == 2 and neg_out.size(1) == 2
            pos_loss = -(pos_label * pos_out.log_softmax(dim=-1)[:, 0]).mean()
            neg_loss = -(neg_out.log_softmax(dim=-1)[:, 1]).mean() if neg_out is not None else 0
        else:
            pos_loss = -(pos_label * pos_out.log()).mean()
            neg_loss = -(neg_out.log()).mean()
    else:
        pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss


def create_input_layer(num_nodes, num_node_feats,
                       use_node_feats=True, node_emb=None):
    emb = None
    if use_node_feats:
        input_dim = num_node_feats
        if node_emb:
            emb = torch.nn.Embedding(num_nodes, node_emb)
            input_dim += node_emb
    else:
        emb = torch.nn.Embedding(num_nodes, node_emb)
        input_dim = node_emb
    return input_dim, emb


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, dropout=0.5, bn=False, activation="elu",
                 use_node_feats=True, num_nodes=None, node_emb=None, random_negative_sampling=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.use_node_feats = use_node_feats
        self.node_emb = node_emb

        if node_emb is not None and num_nodes is None:
            raise RuntimeError("Please provide the argument `num_nodes`.")

        in_channels, self.emb = create_input_layer(
            num_nodes, in_channels, use_node_feats=use_node_feats, node_emb=node_emb
        )
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else out_channels
            second_channels = out_channels

            self.convs.append(GCNConv(first_channels, second_channels))
            self.bns.append(bn(second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

        if random_negative_sampling:
            # this will be faster than pyg negative_sampling
            self.negative_sampler = random_negative_sampler
        else:
            self.negative_sampler = negative_sampling

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

        if self.emb is not None:
            nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, x):
        if self.use_node_feats:
            input_feat = x
            if self.node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def forward(self, x, edge_index, mask=None, temp=(1.,), exclude_layers=None, sparse=True, **kwargs):
        if isinstance(temp, float):
            temp = (temp,)
        if len(temp) == 1:
            temp = list(repeat(temp[0], len(self.convs)))
        if mask is not None:
            return self._lwp_forward(x, edge_index, mask, temp, exclude_layers, sparse, **kwargs)
        else:
            return self._forward(x, edge_index, sparse)

    def _lwp_forward(self, x, edge_index, mask, temp, exclude_layers, sparse, *,
                     recon_loss, decoder, neg_edges, perm):

        batch_veiled_edges = edge_index[:, perm]
        batch_neg_edges = neg_edges[:, perm]
        layer_loss = torch.tensor([], device=x.device)

        def edge_reg(x, bandwidth):
            batch_weight = bandwidth[perm]
            pos_out = decoder(
                x, batch_veiled_edges, sigmoid=False
            )
            neg_out = decoder(x, batch_neg_edges, sigmoid=False)
            return recon_loss(pos_out, neg_out, batch_weight)

        x = self.create_input_feat(x)

        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            bandwidth = mask(edge_index, temp=temp[i])
            if sparse:
                edge_index_sparse = edgeidx2sparse(edge_index, x.size(0), edge_attr=bandwidth)
                x = conv(x, edge_index_sparse)
                x = self.bns[i](x)
                x = self.activation(x)
                if not exclude_layers[i]:
                    layer_loss = torch.cat((layer_loss, edge_reg(x, bandwidth).unsqueeze(-1)), dim=-1)
            else:
                x = conv(x, edge_index, edge_weight=bandwidth)
                x = self.bns[i](x)
                x = self.activation(x)
                if not exclude_layers[i]:
                    layer_loss = torch.cat((layer_loss, edge_reg(x, bandwidth).unsqueeze(-1)), dim=-1)
        x = self.dropout(x)
        bandwidth = mask(edge_index, temp=temp[-1])
        if sparse:
            edge_index_sparse = edgeidx2sparse(edge_index, x.size(0), edge_attr=bandwidth)
            x = self.convs[-1](x, edge_index_sparse)
            x = self.bns[-1](x)
            x = self.activation(x)
            if not exclude_layers[-1]:
                layer_loss = torch.cat((layer_loss, edge_reg(x, bandwidth).unsqueeze(-1)), dim=-1)
        else:
            x = self.convs[-1](x, edge_index, edge_weight=bandwidth)
            x = self.bns[-1](x)
            x = self.activation(x)
            if not exclude_layers[-1]:
                layer_loss = torch.cat((layer_loss, edge_reg(x, bandwidth).unsqueeze(-1)), dim=-1)

        return x, layer_loss

    def _forward(self, x, edge_index, sparse=True):

        x = self.create_input_feat(x)
        if sparse:
            edge_index = edgeidx2sparse(edge_index, x.size(0))

        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)

        return x

    @torch.no_grad()
    def get_embedding(self, x, edge_index, mode="cat", l2_norm=False, sparse=True):

        self.eval()
        assert mode in {"cat", "last"}, mode

        x = self.create_input_feat(x)
        if sparse:
            edge_index = edgeidx2sparse(edge_index, x.size(0))
        out = []
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            out.append(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        out.append(x)

        if mode == "cat":
            embedding = torch.cat(out, dim=1)
        else:
            embedding = out[-1]

        if l2_norm:
            embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed

        return embedding


class DotEdgeDecoder(nn.Module):
    """Simple Dot Product Edge Decoder"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def reset_parameters(self):
        return

    def forward(self, z, edge, sigmoid=True):
        x = z[edge[0]] * z[edge[1]]
        x = x.sum(-1)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class Decoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=False):
        x = z[edge[0]] * z[edge[1]]

        assert x.size(1) == self.mlps[0].in_features

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            return x[:, 0].sigmoid()
        else:
            return x


def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges


class Bandana(nn.Module):
    def __init__(self, encoder, decoder, mask=None, random_negative_sampling=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask = mask
        self.loss_fn = ce_loss

        if random_negative_sampling:
            # this will be faster than pyg negative_sampling
            self.negative_sampler = random_negative_sampler
        else:
            self.negative_sampler = negative_sampling

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def train_epoch(self, data, optimizer, neg_ratio=0.5, temp=(1.,), exclude_layers=[], batch_size=2**16,
                    grad_norm=1.0, sparse=False):

        x, edge_index = data.x, data.edge_index

        loss_total = 0.0
        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=int(data.num_edges * neg_ratio * 2),
        ).view(2, -1)

        for perm in DataLoader(range(data.num_edges), batch_size=batch_size, shuffle=True):

            optimizer.zero_grad()

            z, layer_loss = self.encoder(x, edge_index, mask=self.mask, temp=temp, exclude_layers=exclude_layers,
                                         sparse=sparse, recon_loss=self.loss_fn, decoder=self.decoder,
                                         neg_edges=neg_edges, perm=perm)

            loss = layer_loss.mean(dim=-1)
            loss.backward()

            if grad_norm > 0:
                nn.utils.clip_grad_norm_(self.parameters(), grad_norm)

            optimizer.step()
            loss_total += loss.item()

        return loss_total

    @torch.no_grad()
    def batch_predict(self, z, edges, batch_size=2**16, probing_decoder=None):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            if probing_decoder is not None:
                preds += [probing_decoder(z, edge).squeeze().cpu()]
            else:
                preds += [self.decoder(z, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index, batch_size=2**16, probing_decoder=None):

        pos_pred = self.batch_predict(z, pos_edge_index, batch_size, probing_decoder)
        neg_pred = self.batch_predict(z, neg_edge_index, batch_size, probing_decoder)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))

        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

    @torch.no_grad()
    def test_ogb(self, z, splits, evaluator, batch_size=2**16, probing_decoder=None):

        pos_valid_edge = splits["valid"].pos_edge_label_index
        neg_valid_edge = splits["valid"].neg_edge_label_index
        pos_test_edge = splits["test"].pos_edge_label_index
        neg_test_edge = splits["test"].neg_edge_label_index

        pos_valid_pred = self.batch_predict(z, pos_valid_edge, batch_size, probing_decoder)
        neg_valid_pred = self.batch_predict(z, neg_valid_edge, batch_size, probing_decoder)

        pos_test_pred = self.batch_predict(z, pos_test_edge, batch_size, probing_decoder)
        neg_test_pred = self.batch_predict(z, neg_test_edge, batch_size, probing_decoder)

        results = {}
        for K in [20, 50, 100]:
            evaluator.K = K
            valid_hits = evaluator.eval(
                {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred, }
            )[f"hits@{K}"]
            test_hits = evaluator.eval(
                {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred, }
            )[f"hits@{K}"]

            results[f"Hits@{K}"] = (valid_hits, test_hits)

        return results
