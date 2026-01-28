from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io as sio
import scipy.sparse as sp
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T
import os
from os import path

# 确保 data_utils 存在这些函数
from data_utils import even_quantile_labels, to_sparse_tensor, rand_splits

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, PPI, Reddit
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, homophily, dense_to_sparse


# -------------------------------------------------------------------
# 修复：加载真实 Multiplex 数据集的函数 (ACM, DBLP, IMDB)
# -------------------------------------------------------------------

def load_real_multiplex_dataset(data_dir, dataname, ood_type):
    """
    加载 ACM, DBLP, IMDB 等 .mat 格式的多路复用数据集
    """
    dataname = dataname.lower()
    paths_to_try = [
        f"{data_dir}/{dataname.upper()}/{dataname}.mat",
        f"{data_dir}/{dataname}.mat",
        f"{data_dir}/{dataname.upper()}.mat"
    ]

    raw_data = None
    for p in paths_to_try:
        if path.exists(p):
            try:
                raw_data = sio.loadmat(p)
                print(f"✅ Successfully loaded dataset from: {p}")

                # IMDB 解包逻辑
                if 'imdb' in raw_data:
                    print("📦 Detected nested 'imdb' structure. Unwrapping...")
                    try:
                        struct = raw_data['imdb'][0, 0]
                        for key in struct.dtype.names:
                            raw_data[key] = struct[key]
                        print(f"   -> Unwrapped keys: {struct.dtype.names}")
                    except Exception as e:
                        print(f"   -> ⚠️ Unwrapping failed: {e}")
                break
            except Exception as e:
                print(f"File load error: {e}")
                continue

    if raw_data is None:
        raise FileNotFoundError(f"Could not find {dataname}.mat in {data_dir}")

    # 1. 解析数据
    edge_indices = []

    def get_key(data, keys):
        for k in keys:
            if k in data: return data[k]
        return None

    # 1. 加载特征 x
    x = get_key(raw_data, ['feature', 'features', 'X'])
    if x is None:
        if 'TvsP' in raw_data:
            x = raw_data['TvsP'].transpose()
        elif 'PvsT' in raw_data:
            x = raw_data['PvsT']
        else:
            print("⚠️ No feature found. Using Identity matrix.")
            # 防御性编程：此时 y 还没加载，暂时用 1 占位，后面如果出错会暴露
            x = sp.eye(1)

            # 格式转换
    if sp.issparse(x): x = x.todense()
    if isinstance(x, np.matrix): x = np.asarray(x)
    x = torch.FloatTensor(x)

    # 2. 加载标签 y
    y = get_key(raw_data, ['label', 'labels', 'gnd', 'PvsL', 'Y'])
    if y is None: raise ValueError("Could not find label")

    if sp.issparse(y): y = y.todense()
    if isinstance(y, np.matrix): y = np.asarray(y)
    y = torch.LongTensor(y)

    # 🔴🔴🔴 [安全修复：只针对 IMDB 做特殊处理] 🔴🔴🔴
    # 这个 if 块确保完全不影响 ACM 和 DBLP 的逻辑
    if dataname == 'imdb':
        print(f"🛡️ [IMDB Special Handling] Applying specific fixes for {dataname}...")

        # 1. 修复标签形状 (1, N) -> (N, 1)
        if y.dim() > 1 and y.shape[0] < y.shape[1]:
            print(f"   ⚠️ Transposing labels from {list(y.shape)} to {[y.shape[1], y.shape[0]]}")
            y = y.t()

        # 2. 修复特征形状 (Feat, N) -> (N, Feat)
        # 如果特征行数 != 标签行数，且特征列数 == 标签行数，说明反了
        if x.shape[0] != y.shape[0] and x.shape[1] == y.shape[0]:
            print(f"   ⚠️ Transposing features from {x.shape} to {[x.shape[1], x.shape[0]]}")
            x = x.t()

        # 如果刚才 x 是用 sp.eye(1) 创建的占位符，这里重新创建正确的单位矩阵
        if x.shape[0] != y.shape[0]:
            print(f"   ⚠️ Identity feature matrix size mismatch. Re-creating eye({y.shape[0]})...")
            x = torch.eye(y.shape[0])

        # 3. 特征归一化 (仅 IMDB 执行)
        # 这是解决准确率低的关键步骤
        print(f"   🧹 Performing Row-Normalization on IMDB features...")
        row_sum = x.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        x = x / row_sum
    # 🔴🔴🔴 [结束] 🔴🔴🔴

    # 标准的标签处理 (对所有数据集通用)
    if y.dim() > 1 and y.shape[1] > 1: y = torch.argmax(y, dim=1)
    if y.dim() == 1: y = y.unsqueeze(1)

    found_preprocessed_adj = False
    possible_adjs = ['PLP', 'PAP', 'PSP', 'MAM', 'MDM', 'MGM', 'MKM',
                     'APA', 'APCPA', 'APTPA',
                     'net_APA', 'net_APCPA', 'net_APTPA']
    for key in possible_adjs:
        if key in raw_data:
            print(f"   Found adjacency matrix: {key}")
            edge_indices.append(raw_data[key])
            found_preprocessed_adj = True

    if not found_preprocessed_adj and dataname == 'acm':
        print("Detected RAW ACM data. Constructing Meta-paths...")
        if 'PvsA' in raw_data:
            p_vs_a = sp.csr_matrix(raw_data['PvsA']) if not sp.issparse(raw_data['PvsA']) else raw_data['PvsA']
            edge_indices.append(p_vs_a @ p_vs_a.T)
        if 'PvsC' in raw_data:
            p_vs_c = sp.csr_matrix(raw_data['PvsC']) if not sp.issparse(raw_data['PvsC']) else raw_data['PvsC']
            edge_indices.append(p_vs_c @ p_vs_c.T)
        elif 'PvsP' in raw_data:
            edge_indices.append(raw_data['PvsP'])

    if len(edge_indices) == 0:
        raise ValueError("Could not find any adjacency matrices.")

    tensor_edge_indices = []
    for adj in edge_indices:
        if not sp.issparse(adj): adj = sp.csr_matrix(adj)
        adj.setdiag(0)
        adj.eliminate_zeros()
        adj_coo = adj.tocoo()
        row = torch.from_numpy(adj_coo.row)
        col = torch.from_numpy(adj_coo.col)
        tensor_edge_indices.append(torch.stack([row, col], dim=0).long())

    # 3. 构建 Data 对象
    dataset = Data(x=x, y=y)
    dataset.edge_indices = tensor_edge_indices
    # 🔴 关键：保留一份完整的全图索引，供后续 OOD 分割使用
    full_node_idx = torch.arange(x.shape[0])
    dataset.node_idx = full_node_idx

    train_idx = get_key(raw_data, ['train_idx'])
    val_idx = get_key(raw_data, ['val_idx'])
    test_idx = get_key(raw_data, ['test_idx'])

    if train_idx is not None:
        def to_idx(idx):
            if sp.issparse(idx): idx = idx.todense()
            if isinstance(idx, np.matrix): idx = np.asarray(idx)
            return torch.LongTensor(idx.squeeze())

        dataset.splits = {'train': to_idx(train_idx), 'valid': to_idx(val_idx), 'test': to_idx(test_idx)}
    else:
        dataset.splits = rand_splits(dataset.node_idx, train_prop=.6, valid_prop=.2)

    dataset_ind = dataset
    dataset_ood_tr = None
    dataset_ood_te = None

    # OOD 生成逻辑
    if ood_type == 'structure':
        dataset_ood_tr = create_multiplex_sbm_dataset(dataset)
        dataset_ood_te = create_multiplex_sbm_dataset(dataset)

    elif ood_type == 'feature':
        dataset_ood_tr = create_multiplex_feat_noise_dataset(dataset)
        dataset_ood_te = create_multiplex_feat_noise_dataset(dataset)

    elif ood_type == 'label':
        # 🟢 [修复核心] 针对 Label OOD 的逻辑修正
        num_classes = y.max().item() + 1
        class_t = num_classes - 1

        label = y.squeeze()  # (N,)

        # 定义 IND Mask (类 0 到 T-1)
        center_node_mask_ind = (label < class_t)

        # 定义 OOD Mask (类 T)
        center_node_mask_ood = (label == class_t)

        # 1. 设置 IND 数据集
        # 使用 full_node_idx 来索引，防止 dataset.node_idx 被修改后维度对不上
        dataset_ind.node_idx = full_node_idx[center_node_mask_ind]
        # 重新划分 IND 的训练/验证/测试集 (只在 ID 节点内划分)
        dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=.6, valid_prop=.2)

        # 2. 设置 OOD 数据集
        dataset_ood_tr = Data(x=x, y=y)
        dataset_ood_tr.edge_indices = tensor_edge_indices
        dataset_ood_te = Data(x=x, y=y)
        dataset_ood_te.edge_indices = tensor_edge_indices

        # 使用 full_node_idx 提取 OOD 节点
        dataset_ood_tr.node_idx = full_node_idx[center_node_mask_ood]
        dataset_ood_te.node_idx = full_node_idx[center_node_mask_ood]

    else:
        dataset_ood_tr = dataset
        dataset_ood_te = dataset

    return dataset_ind, dataset_ood_tr, dataset_ood_te


# -------------------------------------------------------------------
# 下面的辅助函数保持不变 (SBM, Feat Noise 等)
# -------------------------------------------------------------------

def create_multiplex_sbm_dataset(data, p_ii=1.5, p_ij=0.5):
    new_edge_indices = []
    n = data.num_nodes
    if data.y.dim() > 1 and data.y.shape[1] > 1:
        num_blocks = data.y.shape[1]
    else:
        num_blocks = int(data.y.max()) + 1
    block_size = n // num_blocks
    block_sizes = [block_size for _ in range(num_blocks - 1)] + [block_size + n % block_size]

    for edge_index in data.edge_indices:
        if n > 1:
            d = edge_index.size(1) / n / (n - 1)
        else:
            d = 0
        current_p_ii, current_p_ij = p_ii * d, p_ij * d
        current_probs = torch.ones((num_blocks, num_blocks)) * current_p_ij
        current_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = current_p_ii
        new_adj = stochastic_blockmodel_graph(block_sizes, current_probs)
        new_edge_indices.append(new_adj)

    dataset = Data(x=data.x, y=data.y)
    dataset.edge_indices = new_edge_indices
    dataset.node_idx = torch.arange(n)
    return dataset


def create_multiplex_feat_noise_dataset(data):
    x = data.x
    n = data.num_nodes
    idx = torch.randint(0, n, (n, 2))
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)
    dataset = Data(x=x_new, y=data.y)
    dataset.edge_indices = data.edge_indices
    dataset.node_idx = torch.arange(n)
    return dataset



def load_dataset(args):
    # 主入口逻辑保持不变
    if args.dataset.lower() in ('acm', 'dblp', 'imdb'):
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_real_multiplex_dataset(args.data_dir, args.dataset,
                                                                                  args.ood_type)
    else:
        raise ValueError(f'Invalid dataname: {args.dataset}')
    return dataset_ind, dataset_ood_tr, dataset_ood_te






def create_sbm_dataset(data, p_ii=1.5, p_ij=0.5):
    new_edge_indices = []
    n = data.num_nodes
    if data.edge_index.size(1) > 0:
        d = data.edge_index.size(1) / data.num_nodes / (data.num_nodes - 1)
    else:
        d = 0
    num_blocks = int(data.y.max()) + 1
    p_ii, p_ij = p_ii * d, p_ij * d
    block_size = n // num_blocks
    block_sizes = [block_size for _ in range(num_blocks - 1)] + [block_size + n % block_size]
    edge_probs = torch.ones((num_blocks, num_blocks)) * p_ij
    edge_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = p_ii
    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)
    dataset = Data(x=data.x, edge_index=edge_index, y=data.y)
    dataset.node_idx = torch.arange(dataset.num_nodes)
    return dataset


def create_feat_noise_dataset(data):
    x = data.x
    n = data.num_nodes
    idx = torch.randint(0, n, (n, 2))
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)
    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)
    return dataset


def create_label_noise_dataset(data):
    y = data.y
    n = data.num_nodes
    idx = torch.randperm(n)[:int(n * 0.5)]
    y_new = y.clone()
    y_new[idx] = torch.randint(0, y.max(), (int(n * 0.5),))
    dataset = Data(x=data.x, edge_index=data.edge_index, y=y_new)
    dataset.node_idx = torch.arange(n)
    return dataset

