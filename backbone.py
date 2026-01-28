import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def intermediate_forward(self, x, edge_index=None, layer_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature_list(self, x, edge_index=None):
        out_list = []
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out_list.append(x)
        x = self.lins[-1](x)
        return x, out_list


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def intermediate_forward(self, x, edge_index, layer_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature_list(self, x, edge_index):
        out_list = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out_list.append(x)
        x = self.convs[-1](x, edge_index)
        return x, out_list


class MultiplexGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 num_relations=1, dropout=0.5, use_bn=True):
        super(MultiplexGCN, self).__init__()

        self.num_relations = num_relations
        self.dropout = dropout

        self.relation_gcns = nn.ModuleList()
        for _ in range(num_relations):
            self.relation_gcns.append(
                GCN(in_channels, hidden_channels, hidden_channels, num_layers, dropout, use_bn=use_bn)
            )

        # 保留最后的分类层
        self.final_lin = nn.Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for gcn in self.relation_gcns:
            gcn.reset_parameters()
        self.final_lin.reset_parameters()

    def forward(self, x, edge_indices):
        """
        x: [N, D]
        edge_indices: List of [2, E_i], length = num_relations
        """
        layer_embeddings = []

        for i, gcn in enumerate(self.relation_gcns):
            # 获取 GCN 输出的 embedding (隐层特征)
            # 注意：这里的 GCN 输出维度是 hidden_channels (因为我们在初始化时把 out_channels 设为了 hidden_channels)
            h = gcn(x, edge_indices[i])
            layer_embeddings.append(h)

        # stack_emb shape: [num_relations, N, hidden]
        stack_emb = torch.stack(layer_embeddings, dim=0)

        # 简单平均 (Mean Pooling)
        fused_emb = stack_emb.mean(dim=0)

        # 最终分类 logits
        logits = self.final_lin(fused_emb)
        return logits