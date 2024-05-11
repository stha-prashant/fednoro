import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

    # basic setting
    parser.add_argument('--exp', type=str,
                        default='Fed', help='experiment name')
    parser.add_argument('--dataset', type=str,
                        default='cifar10', help='dataset name')
    parser.add_argument('--optimizer', type=str,
                        default='adam', help='dataset name')
    parser.add_argument('--model', type=str,
                        default='resnet18', help='model name')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=3e-4,
                        help='base learning rate')
    parser.add_argument('--pretrained', type=int,  default=0)
    parser.add_argument('--num_classes', type=int, default=10, 
                        help="number of classes")

    # for FL
    parser.add_argument('--n_clients', type=int,  default=20,
                        help='number of users') 
    parser.add_argument('--frac', type=float,  default=0.1,
                        help='number of users') 
    parser.add_argument('--iid', type=int, default=0, help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float,
                        default=0.9, help='parameter for non-iid')
    parser.add_argument('--alpha', type=float,
                        default=2.0, help='parameter for non-iid')
    parser.add_argument('--local_ep', type=int, default=1, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=100, help='rounds')

    parser.add_argument('--s1', type=int,  default=10, help='stage 1 rounds')
    parser.add_argument('--begin', type=int,  default=10, help='ramp up begin')
    parser.add_argument('--end', type=int,  default=49, help='ramp up end')
    parser.add_argument('--a', type=float,  default=0.8, help='a')
    parser.add_argument('--warm', type=int,  default=1)
    parser.add_argument('--robust_method', type=str,
                        default='fednoro', help='model name')

    # noise
    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")
    parser.add_argument('--level_n_upperb', type=float, default=1.0, help="upper bound of noise level")
    parser.add_argument('--n_type', type=str, default="instance", help="type of noise")

    parser.add_argument('--wandb',action = 'store_true',
                        help='whether to use wandb')

    
    args = parser.parse_args()
    return args