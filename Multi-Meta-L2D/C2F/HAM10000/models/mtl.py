""" Model for meta-transfer learning. """
import  torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from HAM10000.models.resnet224 import ResNet34


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, depth, hidden_dim, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        # 元阶段换一个分类头，可以跟预训练阶段的维度不一样
        self.out_dim = self.args.meta_classes+self.args.meta_experts
        self.vars = nn.ParameterList()
        if depth == 1:
            w = nn.Parameter(torch.ones([self.out_dim, self.z_dim]))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            b = nn.Parameter(torch.zeros(self.out_dim))
            self.vars.append(b)
        else:
            w = nn.Parameter(torch.ones([self.hidden_dim, self.z_dim]))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            b = nn.Parameter(torch.zeros(self.hidden_dim))
            self.vars.append(b)
            for _ in range(depth - 2):
                w = nn.Parameter(torch.ones([self.hidden_dim, self.hidden_dim]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                b = nn.Parameter(torch.zeros(self.hidden_dim))
                self.vars.append(b)
            w = nn.Parameter(torch.ones([self.out_dim, self.hidden_dim]))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            b = nn.Parameter(torch.zeros(self.out_dim))
            self.vars.append(b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        if self.depth == 1:
            idx = 0
            w = the_vars[idx]
            b = the_vars[idx + 1]
            net = F.linear(input_x, w, b)
            idx += 2
        else:
            idx = 0
            w = the_vars[idx]
            b = the_vars[idx + 1]
            net = F.linear(input_x, w, b)
            net = F.relu(net, inplace=True)
            idx += 2
            for _ in range(self.depth - 2):
                w = the_vars[idx]
                b = the_vars[idx + 1]
                net = F.linear(net, w, b)
                net = F.relu(net, inplace=True)
                idx += 2
            w = the_vars[idx]
            b = the_vars[idx + 1]
            net = F.linear(net, w, b)
            idx += 2
        assert idx == len(the_vars)
        return net

    def parameters(self):
        return self.vars


class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, loss_fn, mode='meta', num_cls=64):
        super().__init__()

        # --base_lr=0.01  '--update_step=50'
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        self.meta_class = args.meta_classes + args.meta_experts

        if self.mode == 'meta':
            # self.encoder = ResNetMtl()
            self.encoder = ResNet34(mtl=True)
            for param in self.encoder.parameters():
                param.requires_grad = False
            param_num = sum(map(lambda x: np.prod(x.shape), self.encoder.parameters()))
            print("encoder Total param", param_num)
        else:
            # self.encoder = ResNetMtl(mtl=False)
            # self.pre_fc = nn.Sequential(nn.Linear(self.encoder.nChannels, 1000), nn.ReLU(), nn.Linear(1000, num_cls))
            self.encoder = ResNet34(mtl=False)
            self.pre_fc = nn.Linear(self.encoder.n_features, args.pre_classes+args.pre_experts)
            self.pre_fc.bias.data.zero_()
            param_num = sum(map(lambda x: np.prod(x.shape), self.encoder.parameters()))
            print("encoder Total param", param_num)
            param_num = sum(map(lambda x: np.prod(x.shape), self.pre_fc.parameters()))
            print("pre_fc Total param", param_num)

        z_dim = self.encoder.n_features
        self.base_learner = BaseLearner(args, args.depth, args.hidden_dim, z_dim)
        param_num = sum(map(lambda x: np.prod(x.shape), self.base_learner.parameters()))
        print("base_learner Total param", param_num)

        self.loss_fn = loss_fn


    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode=='pre':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query, m = inp
            return self.meta_forward(data_shot, label_shot, data_query, m)
        elif self.mode=='preval':
            data_shot, label_shot, data_query, collection_Ms = inp
            return self.preval_forward(data_shot, label_shot, data_query, collection_Ms)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        """The function to forward pretrain phase.
        Args:
          inp: input images.
        Returns:
          the outputs of pretrain model.
        """
        return self.pre_fc(self.encoder(inp))

    def meta_forward(self, data_shot, label_shot, data_query, m):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)

        loss = self.loss_fn(logits, label_shot, m, self.meta_class)

        # loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = self.loss_fn(logits, label_shot, m, self.meta_class)

            # loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)        
        return logits_q

    def preval_forward(self, data_shot, label_shot, data_query, collection_Ms):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)

        loss = self.loss_fn(logits, label_shot, collection_Ms, self.meta_class)
        # loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, 100):
            logits = self.base_learner(embedding_shot, fast_weights)
            # loss = F.cross_entropy(logits, label_shot)
            loss = self.loss_fn(logits, label_shot, collection_Ms, self.meta_class)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)         
        return logits_q

