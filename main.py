import argparse
import random
import numpy as np
import torch
import torch.nn as nn

# 工具类 (保持不变)
from logger import Logger_classify, Logger_detect, save_result
from data_utils import evaluate_classify, evaluate_detect, eval_acc, eval_rocauc, rand_splits
from dataset import load_dataset
from parse import parser_add_main_args

# 🟢 [核心修改] 只导入 EMP 和 GNNSafe，不再导入 baselines
from emp import EMP


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Parse args ###
parser = argparse.ArgumentParser(description='EMP Training Pipeline')
parser_add_main_args(parser)

# 强制设置 method 选项为 emp
for action in parser._actions:
    if action.dest == 'method':
        action.choices = ['emp']

args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load Data ###
dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)

# 标签维度修正
if len(dataset_ind.y.shape) == 1: dataset_ind.y = dataset_ind.y.unsqueeze(1)
if len(dataset_ood_tr.y.shape) == 1: dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
if isinstance(dataset_ood_te, list):
    for data in dataset_ood_te:
        if len(data.y.shape) == 1: data.y = data.y.unsqueeze(1)
else:
    if len(dataset_ood_te.y.shape) == 1: dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

### Get Splits (ACM等数据集必须保留 else 分支) ###
if args.dataset in ['cora', 'citeseer', 'pubmed']:
    pass
else:
    dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
d = dataset_ind.x.shape[1]


# -----------------------------------------------------------------------
# [标准化] 确保数据统一为 Multiplex 格式
# -----------------------------------------------------------------------
def ensure_multiplex_format(data):
    if not hasattr(data, 'edge_indices'):
        data.edge_indices = [data.edge_index]
    return data


dataset_ind = ensure_multiplex_format(dataset_ind)
dataset_ood_tr = ensure_multiplex_format(dataset_ood_tr)
if isinstance(dataset_ood_te, list):
    for i in range(len(dataset_ood_te)): dataset_ood_te[i] = ensure_multiplex_format(dataset_ood_te[i])
else:
    dataset_ood_te = ensure_multiplex_format(dataset_ood_te)

# -----------------------------------------------------------------------
# [GPU 优化]
# -----------------------------------------------------------------------
print("Moving datasets to GPU...")


def move_to_device(data, device):
    if data.x is not None: data.x = data.x.to(device)
    if data.y is not None: data.y = data.y.to(device)
    if hasattr(data, 'edge_indices') and data.edge_indices is not None:
        data.edge_indices = [adj.to(device) for adj in data.edge_indices]
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        data.edge_index = data.edge_index.to(device)
    if hasattr(data, 'node_idx') and torch.is_tensor(data.node_idx):
        data.node_idx = data.node_idx.to(device)
    return data


dataset_ind = move_to_device(dataset_ind, device)
dataset_ood_tr = move_to_device(dataset_ood_tr, device)
if isinstance(dataset_ood_te, list):
    for i in range(len(dataset_ood_te)): dataset_ood_te[i] = move_to_device(dataset_ood_te[i], device)
else:
    dataset_ood_te = move_to_device(dataset_ood_te, device)

# -----------------------------------------------------------------------
# [模型初始化]
# -----------------------------------------------------------------------
num_relations = len(dataset_ind.edge_indices)
print(f"Detected Multiplex Graph with {num_relations} relations.")

if args.method == 'emp':
    if num_relations > 1:
        print(f"Initializing EMP Model...")
        model = EMP(d, c, args, num_relations).to(device)
    else:
        print(f"Warning: Single relation detected. Fallback to standard GNNSafe backbone.")
        model = GNNSafe(d, c, args).to(device)
else:
    raise ValueError("Only 'emp' method is supported.")

### Setup Loss & Metrics ###
if args.dataset in ('proteins', 'ppi'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

if args.dataset in ('proteins', 'ppi', 'twitch'):
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

if args.mode == 'classify':
    logger = Logger_classify(args.runs, args)
else:
    logger = Logger_detect(args.runs, args)

print('MODEL:', model)

### Training Loop ###
for run in range(args.runs):
    # 1. 重置参数
    model.reset_parameters()
    model.to(device)

    # 2. 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_test_results = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
        loss.backward()
        optimizer.step()

        if epoch % args.display_step == 0 or epoch == args.epochs - 1:
            if args.mode == 'classify':
                result = evaluate_classify(model, dataset_ind, eval_func, criterion, args, device)
                logger.add_result(run, result)
                print(
                    f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Test: {100 * result[2]:.2f}%')
            else:
                result = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device)

                # 记录最佳 FPR95
                if best_test_results is None or result[2] < best_test_results[2]:
                    best_test_results = result

                logger.add_result(run, result)

                # 打印所有关键指标: AUROC, AUPR, FPR95, Test Score
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'AUROC: {100 * result[0]:.2f}%, '
                      f'AUPR: {100 * result[1]:.2f}%, '
                      f'FPR95: {100 * result[2]:.2f}%, '
                      f'Test Score: {100 * result[-2]:.2f}%')

    if best_test_results is not None:
        print(
            f"Run {run:02d} Best Results - FPR95: {100 * best_test_results[2]:.2f}%, AUPR: {100 * best_test_results[1]:.2f}%")

    logger.print_statistics(run)

results = logger.print_statistics()

if args.mode == 'detect':
    save_result(results, args)