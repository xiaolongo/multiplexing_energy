import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from backbone import *
import numpy as np

class EMP(nn.Module):
    """
    EMP: Energy Multiplex Propagation
    (原 MultiplexGNNSafe)
    """

    def __init__(self, d, c, args, num_relations):
        super(EMP, self).__init__()

        # Backbone 依然使用 MultiplexGCN
        self.encoder = MultiplexGCN(
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            num_layers=args.num_layers,
            num_relations=num_relations,
            dropout=args.dropout,
            use_bn=args.use_bn
        )
        self.args = args
        self.num_relations = num_relations

        # 初始化层间关系矩阵 M (self.rel_adj)
        if self.args.prop_scheme in ['sequential', 'parallel'] and num_relations > 1:
            self.rel_adj = nn.Parameter(torch.Tensor(num_relations, num_relations))
            nn.init.xavier_uniform_(self.rel_adj)
        else:
            self.rel_adj = None

    def reset_parameters(self):
        self.encoder.reset_parameters()
        if self.rel_adj is not None:
            nn.init.xavier_uniform_(self.rel_adj)

    def _get_processed_adjs(self, dataset, device):
        """缓存归一化的邻接矩阵"""
        if not hasattr(dataset, '_cached_sparse_adjs'):
            dataset._cached_sparse_adjs = []
            N = dataset.x.shape[0]
            for adj in dataset.edge_indices:
                edge_index = adj.to(device)
                row, col = edge_index
                d = degree(col, N).float()
                d_norm = 1. / d[col]
                value = torch.ones_like(row) * d_norm
                value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
                dataset._cached_sparse_adjs.append(adj_t)
        return dataset._cached_sparse_adjs

    def forward(self, dataset, device):
        x = dataset.x.to(device)
        adjs = self._get_processed_adjs(dataset, device)
        return self.encoder(x, adjs)

    def propagation(self, e, adjs, prop_layers=1, alpha=0.5):
        e_in = e.unsqueeze(1)  # [N, 1]

        # 1. Intra-layer Propagation
        intra_propagated_list = []
        for adj in adjs:
            e_temp = e_in.clone()
            for _ in range(prop_layers):
                e_temp = e_temp * alpha + matmul(adj, e_temp) * (1 - alpha)
            intra_propagated_list.append(e_temp.squeeze(1))

        E_intra = torch.stack(intra_propagated_list, dim=1)  # [N, R]

        # 2. Inter-layer Propagation (M Matrix interaction)
        if self.args.prop_scheme == 'intra':
            E_final_matrix = E_intra

        elif self.args.prop_scheme == 'sequential':
            if self.rel_adj is not None:
                E_inter = E_intra.clone()
                for _ in range(prop_layers):
                    # E * M
                    E_inter = E_inter * alpha + torch.matmul(E_inter, self.rel_adj) * (1 - alpha)
                E_final_matrix = E_inter
            else:
                E_final_matrix = E_intra

        elif self.args.prop_scheme == 'parallel':
            if self.rel_adj is not None:
                E_raw = e.unsqueeze(1).repeat(1, self.num_relations)
                E_inter_branch = E_raw.clone()
                for _ in range(prop_layers):
                    E_inter_branch = E_inter_branch * alpha + torch.matmul(E_inter_branch, self.rel_adj) * (1 - alpha)
                E_final_matrix = (E_intra + E_inter_branch) / 2
            else:
                E_final_matrix = E_intra
        else:
            raise ValueError(f"Unknown prop_scheme: {self.args.prop_scheme}")

        return E_final_matrix.mean(dim=1)

    def detect(self, dataset, node_idx, device, args):
        x = dataset.x.to(device)
        adjs = self._get_processed_adjs(dataset, device)
        logits = self.encoder(x, adjs)

        if args.dataset in ('proteins', 'ppi'):
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else:
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)

        if args.use_prop:
            neg_energy = self.propagation(neg_energy, adjs, args.K, args.alpha)

        return neg_energy[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        x_in = dataset_ind.x.to(device)
        adjs_in = self._get_processed_adjs(dataset_ind, device)
        x_out = dataset_ood.x.to(device)
        adjs_out = self._get_processed_adjs(dataset_ood, device)

        logits_in = self.encoder(x_in, adjs_in)
        logits_out = self.encoder(x_out, adjs_out)

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        # Classification Loss
        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        # Regularization Loss (Energy Margin)
        if args.use_reg:
            energy_in = -args.T * torch.logsumexp(logits_in / args.T, dim=-1)
            energy_out = -args.T * torch.logsumexp(logits_out / args.T, dim=-1)

            if args.use_prop:
                energy_in = self.propagation(energy_in, adjs_in, args.K, args.alpha)[train_in_idx]
                energy_out = self.propagation(energy_out, adjs_out, args.K, args.alpha)[train_ood_idx]
            else:
                energy_in = energy_in[train_in_idx]
                energy_out = energy_out[train_in_idx]

            # Shape matching
            if energy_in.shape[0] != energy_out.shape[0]:
                min_n = min(energy_in.shape[0], energy_out.shape[0])
                energy_in = energy_in[:min_n]
                energy_out = energy_out[:min_n]

            reg_loss = torch.mean(F.relu(energy_in - args.m_in) ** 2 + F.relu(args.m_out - energy_out) ** 2)
            loss = sup_loss + args.lamda * reg_loss
        else:
            loss = sup_loss

        return loss
