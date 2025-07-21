import random
from itertools import combinations

import numpy as np


class synth_expert2:
    def __init__(self, classes_oracle, n_classes=10, p_in=1.0, p_out=0.1):
        '''
        class to model the non-overlapping synthetic experts

        The expert predicts correctly for classes k1 (inclusive) to k2 (exclusive), and
        random across the total number of classes for other classes outside of [k1, k2).

        For example, an expert could be correct for classes 2 (k1) to 4 (k2) for CIFAR-10.

        '''
        self.classes_oracle = classes_oracle
        self.p_in = p_in
        self.p_out = p_out
        self.n_classes = n_classes

    # expert correct in [k1, k2) classes with prob. p_in; correct on other classes with prob. p_out
    def __call__(self, input, labels, label_sparse=None):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i].item() == self.classes_oracle:
                coin_flip = np.random.binomial(1, self.p_in)
                if coin_flip == 1:
                    outs[i] = labels[i].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
            else:
                coin_flip = np.random.binomial(1, self.p_out)
                if coin_flip == 1:
                    outs[i] = labels[i].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
        return outs


# expert correct in class_oracle with prob. p_in; correct on other classes with prob. p_out
class SyntheticExpertOverlap():
    def __init__(self, classes_oracle, n_classes=10, p_in=1.0, p_out=0.1):
        self.expert_static = True
        self.classes_oracle = classes_oracle
        if isinstance(self.classes_oracle, int):
            self.classes_oracle = [self.classes_oracle]
        # if self.class_oracle is None:
        #     self.class_oracle = random.randint(0, n_classes-1)
        #     self.expert_static = False
        self.n_classes = n_classes
        self.p_in = p_in
        self.p_out = p_out

    # def resample(self):
    #     if not self.expert_static:
    #         self.class_oracle = random.randint(0, self.n_classes-1)

    def __call__(self, images, labels, labels_sparse=None):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i].item() in self.classes_oracle:
                coin_flip = np.random.binomial(1, self.p_in)
                if coin_flip == 1:
                    outs[i] = labels[i].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes-1)
            else:
                coin_flip = np.random.binomial(1, self.p_out)
                if coin_flip == 1:
                    outs[i] = labels[i].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes-1)
        return outs
