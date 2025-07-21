import sys
import logging
import os
import argparse
import torch
import random
import numpy as np

from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.meta import MetaTrainer
from trainer.pre import PreTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='organcmnist',
                        choices=["organamnist", "organcmnist", "organsmnist"])
    parser.add_argument('--test_dataset', type=str, default='organamnist',
                        choices=["organamnist", "organcmnist", "organsmnist"])
    parser.add_argument('--phase', type=str, default='meta_train',
                        choices=['pre_train', 'meta_train', 'meta_eval'])
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--loss_type', type=str, default='softmax',
                        choices=['softmax', 'ova'])
    parser.add_argument('--norm_type', type=str, default="frn")

    ## MedMNIST2D
    parser.add_argument('--size', type=int, default=28,
                        help='the image size of the dataset, 28 or 64 or 128 or 224, default=28')
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize', help='resize images of size 28x28 to 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb', help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_flag', type=str, default='resnet18',
                        help='choose backbone from resnet18, resnet50')

    # Parameters for meta-train phase
    parser.add_argument('--meta_epochs', type=int, default=100)  # Epoch number for meta-train phase
    parser.add_argument('--meta_experts', type=int, default=3)
    parser.add_argument('--meta_batch_size', type=int, default=128)
    parser.add_argument('--meta_val_batch_size', type=int, default=8)
    parser.add_argument('--meta_test_batch_size', type=int, default=64)
    parser.add_argument('--meta_lr1', type=float, default=0.0001)  # Learning rate for SS weights
    parser.add_argument('--meta_lr2', type=float, default=0.001)  # Learning rate for FC weights
    parser.add_argument('--base_lr', type=float, default=0.01)  # Learning rate for the inner loop
    parser.add_argument('--update_step', type=int, default=10)  # The number of updates for the inner loop
    parser.add_argument('--step_size', type=int, default=5)  # The number of epochs to reduce the meta learning rates
    parser.add_argument('--gamma', type=float, default=0.5)  # Gamma for the meta-train learning rate decay
    parser.add_argument('--init_weights', type=str, default=None)  # The pre-trained weights for meta-train phase
    parser.add_argument('--eval_weights', type=str, default=None)  # The meta-trained weights for meta-eval phase
    parser.add_argument('--meta_label', type=str, default='default')  # Additional label for meta-train

    # Baselearner
    parser.add_argument('--depth', type=int, default=3)         #@@ 1-5
    parser.add_argument('--hidden_dim', type=int, default=256)  #@@ 32.64,128,256

    # Parameters for pretain phase
    parser.add_argument('--pre_epochs', type=int, default=100)
    parser.add_argument('--pre_experts', type=int, default=8)
    parser.add_argument('--pre_batch_size', type=int, default=128)
    parser.add_argument('--pre_lr', type=float, default=0.01)
    parser.add_argument('--pre_gamma', type=float, default=0.2)
    parser.add_argument('--pre_step_size', type=int, default=30)
    parser.add_argument('--pre_custom_momentum', type=float, default=0.9)
    parser.add_argument('--pre_custom_weight_decay', type=float, default=0.0005)

    # Context dataset parameters
    parser.add_argument('--n_cntx_pts', type=int, default=50)
    parser.add_argument('--p_out', type=float, default=0.1)

    # Save 250120_043222
    parser.add_argument('--log_dir', type=str, default='./logs')  # Additional label for meta-train
    parser.add_argument('--time_stamp_pre', type=str, default="250122_100143")
    parser.add_argument('--time_stamp_meta', type=str, default=None)

    # Set and print the parameters
    args = parser.parse_args()
    set_seed(args.seed)


    pprint(vars(args))

    # Start trainer for pre-train, meta-train or meta-eval
    if args.phase=='meta_train':
        trainer = MetaTrainer(args)
        trainer.train()
        trainer.eval()
    elif args.phase=='meta_eval':
        trainer = MetaTrainer(args)
        trainer.eval()
    elif args.phase=='pre_train':
        trainer = PreTrainer(args)
        trainer.train()
    else:
        raise ValueError('Please set the correct phase.')
