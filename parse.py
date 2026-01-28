import argparse


def parser_add_main_args(parser):
    # -----------------------------------------------------------------------
    # 1. Setup and Protocol (Basic dataset and environment settings)
    # -----------------------------------------------------------------------
    parser.add_argument('--dataset', type=str, default='cora',
                        help='dataset name: cora, acm, dblp, imdb, etc.')
    parser.add_argument('--ood_type', type=str, default='structure', choices=['structure', 'label', 'feature'],
                        help='Type of OOD data. For Multiplex graphs like ACM/DBLP, usually "label" or "structure".')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_prop', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.1,
                        help='validation label proportion')
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=200)

    # -----------------------------------------------------------------------
    # 2. Model Architecture (EMP / Backbone settings)
    # -----------------------------------------------------------------------
    # Only 'emp' is supported now
    parser.add_argument('--method', type=str, default='emp', choices=['emp'],
                        help='Method name. Only EMP is supported.')

    parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'mlp'],
                        help='Backbone encoder type (default: gcn). Note: MultiplexGCN uses GCN internally.')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for GNN backbone')

    # GAT heads (kept for compatibility if you switch backbone later)
    parser.add_argument('--gat_heads', type=int, default=8, help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1, help='out heads for gat')

    # -----------------------------------------------------------------------
    # 3. EMP Specific Hyperparameters (Energy & Propagation)
    # -----------------------------------------------------------------------
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax/Energy calculation')

    # Regularization (Energy Margin Loss)
    parser.add_argument('--use_reg', action='store_true', help='whether to use energy regularization loss')
    parser.add_argument('--lamda', type=float, default=1.0, help='weight for regularization loss')
    parser.add_argument('--m_in', type=float, default=-5, help='upper bound for in-distribution energy (margin)')
    parser.add_argument('--m_out', type=float, default=-1, help='lower bound for OOD energy (margin)')

    # Propagation (EMP Core)
    parser.add_argument('--use_prop', action='store_true', help='whether to use energy propagation')
    parser.add_argument('--K', type=int, default=2, help='number of propagation steps (layers)')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual connection')

    # 🟢 Propagation Scheme (The key innovation)
    # 'intra': Baseline (Intra-layer only)
    # 'sequential': Intra -> Inter (Layer interaction)
    # 'parallel': Intra || Inter -> Mean (Parallel pathways)
    parser.add_argument('--prop_scheme', type=str, default='intra',
                        choices=['intra', 'sequential', 'parallel'],
                        help='EMP propagation scheme: intra, sequential, or parallel')

    # -----------------------------------------------------------------------
    # 4. Training Optimization
    # -----------------------------------------------------------------------
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_bn', action='store_true', help='use batch normalization')

    # -----------------------------------------------------------------------
    # 5. Display and Utility
    # -----------------------------------------------------------------------
    parser.add_argument('--display_step', type=int, default=1, help='how often to print training status')
    parser.add_argument('--cached', action='store_true', help='cache adjacency matrix for speed (optional)')
    parser.add_argument('--print_prop', action='store_true', help='print proportions of predicted class')
    parser.add_argument('--print_args', action='store_true', help='print args for hyper-parameter searching')
    parser.add_argument('--mode', type=str, default='detect', choices=['classify', 'detect'],
                        help='Training mode: classify or detect')