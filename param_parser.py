"""Parsing the parameters."""

import argparse

def parameter_args():

    parser = argparse.ArgumentParser(description="Run GWNN.")

    parser.add_argument('--data_dir',
                        type=str,
                        default="../data/",
                        help='Path at which to store PyTorch Geometric datasets and look for precomputed files.')

    parser.add_argument("--log-path",
                        nargs="?",
                        default="/home/liuruyue/code/filter/{}_{}_{}.txt")

    parser.add_argument("--epochs",
                        type=int,
                        default=500,
	                help="Number of training epochs. Default is 200.")

    parser.add_argument("--max_initial_scale",
                        type=float,
                        default=5.0,
                        help="Maximum value of the scale. Default is 5.0.")

    parser.add_argument("--num_bands",
                        type=int,
                        default=3,
                        help="Number of bandpass filters. Default is 3.")

    parser.add_argument("--num_scales",
                        type=int,
                        default=5,
                        help="Number of scales. Default is 10.")

    parser.add_argument("--S",
                        type=int,
                        default=20,
                        help="Number of eigenvalue groupings. Default is 10.")

    parser.add_argument("--R",
                        type=int,
                        default=10,
                        help="Number of rademacher samples. Default is 10.")

    parser.add_argument("--alpha",
                        type=float,
                        default=0.8,
                        help="The ratio of the features. Default is 0.8.")

    parser.add_argument("--gama",
                        type=float,
                        default=1,
                        help="The ratio of the graph wavelet term. Default is 0.4.")

    parser.add_argument("--feature_mask",
                        type=float,
                        default=0.2,
                        help="Drop feature ratio of the augmentation. Default is 200.")

    parser.add_argument("--temp",
                        type=float,
                        default=1.0,
                        help="Temperature. Default is 1.0.")


    parser.add_argument("--k",
                        type=int,
                        default=3,
	                help="Order of polynomial. Default is 3.")


    parser.add_argument("--dropout",
                        type=float,
                        default=0.2,
	                help="Dropout probability. Default is 0.2.")


    parser.add_argument("--cfg",
                        type=list,
                        default=[256,128],
                        help="Dropout probability. Default is 0.2.")
    

    parser.add_argument("--seed",
                        type=int,
                        default=500,
	                help="Random seed for sklearn pre-training. Default is 50.")



    parser.add_argument("--train_ratio",
                        type=float,
                        default=0.8,
                        help="The ratio of train data. Default is 0.8.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.01.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-3,
	                help="Adam weight decay. Default is 10^-5.")

    parser.add_argument("--xuhao",
                        type=int,
                        default=1,
                        help="Adam weight decay. Default is 10^-5.")

    return parser.parse_args()
