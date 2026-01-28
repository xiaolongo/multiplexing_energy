import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import time
import os

from logger import Logger_classify, Logger_detect
# 🟢 只导入必要的评估函数
from data_utils import evaluate_classify, evaluate_detect, eval_acc, eval_rocauc, rand_splits
from dataset import load_dataset
from parse import parser_add_main_args
# 🟢 只导入 EMP 模型
from emp import EMP


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Parse args ###
parser = argparse.ArgumentParser(description='EMP Analysis Pipeline (Hyper-parameter, Efficiency, Visualization)')
parser_add_main_args(parser)
parser.add_argument('--dis_type', type=str, default='margin',
                    choices=['margin', 'lamda', 'prop', 'backbone', 'time', 'vis_energy'])

# 强制将 method 设为 emp
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

# 获取数据划分
if args.dataset in ['cora', 'citeseer', 'pubmed']:
    pass
else:
    dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
d = dataset_ind.x.shape[1]

print(f"Discussion of {args.dis_type} on dataset {args.dataset}")


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
# [模型加载] 只加载 EMP
# -----------------------------------------------------------------------
num_relations = len(dataset_ind.edge_indices)
print(f"Detected Multiplex Graph with {num_relations} layers.")

if num_relations > 1:
    print(f"Initializing EMP Model for {num_relations} relations...")
    model = EMP(d, c, args, num_relations).to(device)
else:
    # 如果是单图，EMP 也可以兼容（或者你需要在这里fallback到单图版）
    print("Warning: Single relation detected. Initializing EMP (Single Mode)...")
    model = EMP(d, c, args, num_relations=1).to(device)

### Loss Function ###
if args.dataset in ('proteins', 'ppi'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Metric ###
if args.dataset in ('proteins', 'ppi', 'twitch'):
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

### Logger ###
if args.mode == 'classify':
    logger = Logger_classify(args.runs, args)
else:
    logger = Logger_detect(args.runs, args)

model.train()
print('MODEL:', model)

val_loss_min = 100.
train_time = 0

### Training loop ###
for run in range(args.runs):
    model.reset_parameters()
    # 🟢 修复：在 loop 内定义 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        train_start = time.time()
        loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
        loss.backward()
        optimizer.step()
        train_time += time.time() - train_start

        if args.mode == 'classify':
            result = evaluate_classify(model, dataset_ind, eval_func, criterion, args, device)
            logger.add_result(run, result)

            if epoch % args.display_step == 0:
                print(
                    f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Test: {100 * result[2]:.2f}%')
        else:
            # 🟢 使用 return_score=True 获取具体的能量值
            result, test_in_score, test_ood_score = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion,
                                                                    eval_func, args, device, return_score=True)
            logger.add_result(run, result)

            # 记录最佳模型用于可视化
            if result[-1] < val_loss_min:
                val_loss_min = result[-1]
                in_score, ood_score = test_in_score, test_ood_score

            if epoch % args.display_step == 0:
                print(
                    f'Epoch: {epoch:02d}, Loss: {loss:.4f}, AUROC: {100 * result[0]:.2f}%, FPR95: {100 * result[2]:.2f}%')

    logger.print_statistics(run)

results = logger.print_statistics()

# 如果是时间分析，进行推理时间测试
if args.dis_type == 'time':
    infer_start = time.time()
    with torch.no_grad():
        test_ind_score = model.detect(dataset_ind, dataset_ind.splits['test'], device, args)
    infer_time = time.time() - infer_start

### Save results ###
if not os.path.exists(f'results/discuss'):
    os.makedirs(f'results/discuss')
if not os.path.exists(f'results/vis_scores'):
    os.makedirs(f'results/vis_scores')

# 1. 能量可视化数据保存
if args.dis_type == 'vis_energy':
    if args.use_prop:
        name = 'emp++' if args.use_reg else 'emp'
    else:
        name = 'emp++ w/o prop' if args.use_reg else 'emp w/o prop'

    filename = 'results/vis_scores/' + name + '.csv'
    print(f"Saving visualization scores to {filename}")

    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{in_score.shape[0]} {ood_score.shape[0]}\n")
        for i in range(in_score.shape[0]):
            write_obj.write(f"{in_score[i]}\n")
        for i in range(ood_score.shape[0]):
            write_obj.write(f"{ood_score[i]}\n")

# 2. 其他讨论数据保存 (Margin, Lamda, Time 等)
else:
    filename = f'results/discuss/{args.dis_type}.csv'
    print(f"Saving discussion results to {filename}")

    with open(f"{filename}", 'a+') as write_obj:
        if args.dis_type == 'time':
            write_obj.write(f"EMP {args.dataset} {args.ood_type} {train_time:.4f} {infer_time:.4f}\n")
        else:
            # 记录超参数配置
            if args.dis_type == 'margin':
                write_obj.write(f"{args.dataset} {args.ood_type} {args.m_in} {args.m_out}\n")
            elif args.dis_type == 'prop':
                write_obj.write(f"{args.dataset} {args.ood_type} {args.K} {args.alpha}\n")
            elif args.dis_type == 'lamda':
                write_obj.write(f"{args.dataset} {args.ood_type} {args.lamda}\n")
            elif args.dis_type == 'backbone':
                write_obj.write(f"{args.dataset} {args.ood_type} {args.backbone}\n")

            # 记录指标 (AUROC, AUPR, FPR)
            for k in range(results.shape[1] // 3):
                r = results[:, k * 3]
                write_obj.write(f'OOD Test {k + 1} Final AUROC: {r.mean():.2f} ± {r.std():.2f} ')
                r = results[:, k * 3 + 1]
                write_obj.write(f'OOD Test {k + 1} Final AUPR: {r.mean():.2f} ± {r.std():.2f} ')
                r = results[:, k * 3 + 2]
                write_obj.write(f'OOD Test {k + 1} Final FPR: {r.mean():.2f} ± {r.std():.2f}\n')

            # 记录 IND Score
            r = results[:, -1]
            write_obj.write(f'IND Test Score: {r.mean():.2f} ± {r.std():.2f}\n')