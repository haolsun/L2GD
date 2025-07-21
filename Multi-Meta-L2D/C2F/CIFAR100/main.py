# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NOTE: library code for MTL
# - https://github.com/yaoyao-liu/meta-transfer-learning
# Copyright (c) 2019, Yaoyao Liu.
# All rights reserved.
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Main function for this repo. """
import sys
import logging
import os
import argparse
import torch
import random
import numpy as np

from CIFAR100.utils.misc import pprint
from CIFAR100.utils.gpu_tools import set_gpu
from CIFAR100.trainer.meta import MetaTrainer
from CIFAR100.trainer.pre import PreTrainer


device = torch.device("cuda:0")

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
    parser.add_argument('--seed', type=int, default=0)  # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet'])  # The network architecture
    parser.add_argument('--dataset', type=str,
                        default='cifar100_SC', choices=['cifar100_SC'])  # Dataset
    parser.add_argument('--phase', type=str, default='pre_train',
                        choices=['pre_train', 'meta_train', 'meta_eval'])  # Phase
    parser.add_argument('--loss_type', type=str, default='softmax', choices=['softmax', 'ova'])
    parser.add_argument('--normlayer', type=str, default="frn")

    # Parameters for meta-train phase
    parser.add_argument('--max_epoch', type=int, default=100)  # Epoch number for meta-train phase
    parser.add_argument('--meta_batch_size', type=int, default=128)
    parser.add_argument('--meta_val_batch_size', type=int, default=8)
    parser.add_argument('--meta_test_batch_size', type=int, default=1)
    parser.add_argument('--meta_classes', type=int, default=20)  # Way number, how many classes in a task
    parser.add_argument('--meta_experts', type=int, default=8)  # Way number, how many classes in a task
    parser.add_argument('--meta_lr1', type=float, default=0.0001)  # Learning rate for SS weights
    parser.add_argument('--meta_lr2', type=float, default=0.001)  # Learning rate for FC weights
    parser.add_argument('--base_lr', type=float, default=0.01)  # Learning rate for the inner loop
    parser.add_argument('--update_step', type=int, default=10)  # The number of updates for the inner loop
    parser.add_argument('--step_size', type=int, default=5)  # The number of epochs to reduce the meta learning rates
    parser.add_argument('--gamma', type=float, default=0.5)  # Gamma for the meta-train learning rate decay
    parser.add_argument('--init_weights', type=str, default=None)  # The pre-trained weights for meta-train phase
    parser.add_argument('--eval_weights', type=str, default=None)  # The meta-trained weights for meta-eval phase
    parser.add_argument('--depth', type=int, default=3)  # Additional label for meta-train
    parser.add_argument('--hidden_dim', type=int, default=512)  # Additional label for meta-train
    parser.add_argument('--meta_label', type=str, default='default')  # Additional label for meta-train
    parser.add_argument('--meta_task_size', type=int, default=4)  # Additional label for meta-train

    # Parameters for pretain phase
    parser.add_argument('--pre_max_epoch', type=int, default=200) # Epoch number for pre-train phase
    parser.add_argument('--pre_classes', type=int, default=20)  # Way number, how many classes in a task
    parser.add_argument('--pre_experts', type=int, default=10)  # Way number, how many classes in a task
    parser.add_argument('--pre_batch_size', type=int, default=128) # Batch size for pre-train phase
    parser.add_argument('--pre_lr', type=float, default=0.1) # Learning rate for pre-train phase
    parser.add_argument('--pre_gamma', type=float, default=0.2) # Gamma for the pre-train learning rate decay
    parser.add_argument('--pre_step_size', type=int, default=30) # The number of epochs to reduce the pre-train learning rate
    parser.add_argument('--pre_custom_momentum', type=float, default=0.9) # Momentum for the optimizer during pre-train
    parser.add_argument('--pre_custom_weight_decay', type=float, default=0.0005) # Weight decay for the optimizer during pre-train

    # Context dataset parameters
    parser.add_argument('--n_cntx_pts', type=int, default=100)
    parser.add_argument('--p_out', type=float, default=0.1)
    parser.add_argument('--name', type=str, default="cifar100_SC_pre")

    # Set and print the parameters
    args = parser.parse_args()
    args.device = device
    set_seed(args.seed)
    args.meta_label = 'exp' + str(args.meta_experts) + '_depth' + str(args.depth) + '_dim' + str(args.hidden_dim) + '_seed'+str(args.seed)

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
        raise ValueError('Please set correct phase.')
