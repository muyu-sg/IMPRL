
import argparse
import sys


## Parameter setting
def parameter_parser():
    parser = argparse.ArgumentParser()

    current_dir = sys.path[0]
    parser.add_argument("--path", type=str, default=current_dir)

    parser.add_argument("--data_path", type=str, default="/data/lansy/GCN_Fuse/data/", help="Path of datasets.")

    parser.add_argument("--task", type=str, default="clustering", help="clustering or semiClassifier")

    parser.add_argument("--input_type", type=str, default="feature", help="feature or similarity")
    # input_type； choose features or similarity graphs to learn a multi-variate heterogeneous representation.

    parser.add_argument("--fusion_type", type=str, default="average", help="Fusion Methods: trust")
    
    parser.add_argument("--active", type=str, default="l1", help="l21 or l1")
    # the type of regularizer with Prox_h()

    parser.add_argument("--device", default="3", type=str, required=False)
    parser.add_argument("--fix_seed", action='store_true', default=True, help="")
    parser.add_argument("--seed", type=int, default=40, help="Random seed, default is 42.")
    parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument("--ratio", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--weight_decay", type=float, default=0.15, help="Weight decay")
    parser.add_argument("--in_size", type=int, default=32)

    parser.add_argument("--hdim", nargs='+', type=int, default=[128], help="Number of hidden dimensions") # hyper

    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--gamma', type=float, default=10000)
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--block', type=int, default=1, help='block') # for the example dataset, block can set 2 and more than 2
    parser.add_argument('--thre', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--lamb', type=float, default=1, help='lambda')
    parser.add_argument('--alpha', type=float, default=1, help='alpha')

    return parser.parse_args()