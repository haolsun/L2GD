""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import random
import time

from CIFAR100.dataloader.samplers import CategoriesSampler
from CIFAR100.models.mtl import MtlLearner
from CIFAR100.utils.misc import Averager, Timer, ensure_path
from CIFAR100.utils.misc import count_acc_sys_multiexp, count_acc_sys, count_acc, count_acc_l2d
from CIFAR100.dataloader.dataset_loader import DatasetLoader as Dataset
from CIFAR100.dataloader.dataset_loader import CIFAR10_Sampled
from CIFAR100.lib.dataset import load_cifar, ContextSampler
from CIFAR100.lib.experts import Cifar20SyntheticExpert
from CIFAR100.lib.losses import reject_CrossEntropyLoss, multi_reject_CrossEntropyLoss
from CIFAR100.utils.misc import get_logger
from CIFAR100.utils.misc import multi_exp_evaluate, evaluate, evaluate_data
from CIFAR100.utils.misc import AverageMeter, accuracy


class PreTrainer(object):
    """The class that contains the code for the pretrain phase."""

    def __init__(self, args):

        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'pre')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)

        save_path1 = '_'.join([args.dataset, args.model_type])
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(
            args.pre_gamma) + '_step' + \
                     str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch) + '_exp' + str(args.name)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        self.logger = get_logger(os.path.join(args.save_path, "pre_train.log"))

        self.args = args
        self.device = args.device
        self.pre_classes = args.pre_classes
        self.pre_experts = args.pre_experts
        self.meta_classes = args.meta_classes
        self.meta_experts = args.meta_experts


        sample_expert = 10  # assume exactly divisible by 2
        pop_dict = {"pop1": [i for i in range(0, 10)],
                    "pop2": [i for i in range(10, 20)]}
        self.experts_test = []
        self.experts_train = []
        can_class_train = []
        can_class_test = []
        n_oracle_superclass = 4
        n_oracle_subclass = 3  # 3 or 4 here. Affects gap between {single,pop,pop_attn}
        p_in = 1.0
        for pop_k, pop_v in pop_dict.items():
            train_exp = []
            for _ in range(sample_expert):  # train
                classes_coarse = np.random.choice(pop_v, n_oracle_superclass, replace=False)
                expert = Cifar20SyntheticExpert(classes_coarse, n_classes=self.meta_classes, p_in=1.0,
                                                p_out=args.p_out, \
                                                n_oracle_subclass=n_oracle_subclass)
                can_class_train.append(classes_coarse)
                train_exp.append(expert)
            self.experts_train += train_exp
            self.experts_test += train_exp[
                            :sample_expert // 2]  # pick 50% experts from experts_train (order not matter)

        for pop_k, pop_v in pop_dict.items():
            for _ in range(sample_expert // 2):  # then sample 50% new experts
                classes_coarse = np.random.choice(pop_v, size=n_oracle_superclass, replace=False)
                expert = Cifar20SyntheticExpert(classes_coarse, n_classes=self.meta_classes, p_in=1.0,
                                                p_out=args.p_out, \
                                                n_oracle_subclass=n_oracle_subclass)
                can_class_test.append(classes_coarse)
                self.experts_test.append(expert)
        self.experts_list = np.random.choice(self.experts_train, self.pre_experts).tolist()


        train_data, val_data, test_data = load_cifar(variety='20_100', data_aug=True, seed=args.seed)

        kwargs = {'num_workers': 0, 'pin_memory': True}

        self.train_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=args.pre_batch_size, shuffle=True,
                                                        **kwargs)  # drop_last=True
        self.valid_loader = torch.utils.data.DataLoader(val_data,
                                                        batch_size=args.pre_batch_size, shuffle=False,
                                                        **kwargs)
        # Context sampler val/test-time: partition val/test sets
        prop_cntx = 0.2
        val_cntx_size = int(prop_cntx * len(val_data))
        val_data_cntx, val_data_trgt = torch.utils.data.random_split(val_data,
                                                                     [val_cntx_size, len(val_data) - val_cntx_size], \
                                                                     generator=torch.Generator().manual_seed(
                                                                         args.seed))
        self.cntx_sampler_val = ContextSampler(images=val_data_cntx.dataset.data[val_data_cntx.indices],
                                               labels=val_data_cntx.dataset.targets[val_data_cntx.indices],
                                               transform=val_data.transform,
                                               labels_sparse=val_data_cntx.dataset.targets_sparse[
                                                   val_data_cntx.indices] if
                                               args.dataset == 'cifar20_100' else None,
                                               n_cntx_pts=args.n_cntx_pts, device=self.device, **kwargs)
        self.meta_valid_loader = torch.utils.data.DataLoader(val_data_trgt,
                                                             batch_size=64, shuffle=False,
                                                             **kwargs)  # shuffle=True, drop_last=True


        self.pre_loss_fn = multi_reject_CrossEntropyLoss
        self.meta_loss_fn = multi_reject_CrossEntropyLoss


        self.pre_outdim = self.pre_classes + self.pre_experts
        self.model = MtlLearner(self.args, self.meta_loss_fn, mode='pre', num_cls=self.pre_outdim)
        param_num = sum(map(lambda x: np.prod(x.shape), self.model.parameters()))
        self.logger.info("Total param" + str(param_num))


        self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.pre_lr}, \
                                          {'params': self.model.pre_fc.parameters(), 'lr': self.args.pre_lr}], \
                                         momentum=self.args.pre_custom_momentum, nesterov=True,
                                         weight_decay=self.args.pre_custom_weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, len(self.train_loader) * args.pre_max_epoch)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.to(self.device)

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['val_loss'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0


        best_validation_loss = np.inf
        for epoch in range(1, self.args.pre_max_epoch + 1):

            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()

            self.model.train()
            self.model.mode = 'pre'

            end = time.time()
            epoch_train_loss = []

            # Using tqdm to read samples from train loader
            for i, batch in enumerate(self.train_loader):

                if len(batch) == 2:
                    data, label = [_.to(self.device) for _ in batch]
                    label_sparse = None
                else:
                    data, label, label_sparse = [_.to(self.device) for _ in batch]  # ignore additional labels

                # Output logits for model
                logits = self.model(data)
                # Calculate train loss
                collection_Ms = []
                expert_predictions = []
                for expert in self.experts_list:
                    exp_preds = torch.tensor(expert(data, label, label_sparse),
                                             device=self.device)
                    m = (exp_preds == label).int()
                    collection_Ms.append(m)
                    expert_predictions.append(exp_preds)

                loss = self.pre_loss_fn(logits, label, collection_Ms, self.pre_outdim, with_softmax=True)
                outputs = F.softmax(logits, dim=1)
                epoch_train_loss.append(loss.item())

                # measure accuracy and record loss
                prec1 = accuracy(outputs[:, :self.pre_classes].data, label, topk=(1,))[0]
                losses.update(loss.data.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 50 == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(self.train_loader), batch_time=batch_time,
                        loss=losses, top1=top1), flush=True)

            self.model.eval()
            # self.model.mode = 'preval'
            metrics = self.evaluate(self.valid_loader)

            self.model.mode = 'preval'
            val_metrics = self.validate()
            # Update best saved model
            if val_metrics['system_accuracy'] > trlog['max_acc']:
                trlog['max_acc'] = val_metrics['system_accuracy']
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 50 == 0:
                self.save_model('epoch' + str(epoch))


    def evaluate(self, data_loader):
        '''
        Computes metrics for deferal
        -----
        Arguments:
        net: model
        expert_fn: expert model
        n_classes: number of classes
        loader: data loader
        '''
        correct = 0
        correct_sys = 0
        exp = 0
        exp_total = 0
        total = 0
        real_total = 0
        alone_correct = 0
        #  === Individual Expert Accuracies === #
        expert_correct_dic = {k: 0 for k in range(len(self.experts_list))}
        expert_total_dic = {k: 0 for k in range(len(self.experts_list))}
        #  === Individual  Expert Accuracies === #
        all_classes = self.pre_classes + len(self.experts_list)

        losses = []
        self.model.eval()  # Crucial for networks with batchnorm layers!
        with torch.no_grad():
            for data in data_loader:
                if len(data) == 2:
                    images, labels = data
                else:
                    images, labels, labels_sparse = data  # ignore additional labels
                    labels_sparse = labels_sparse.to(self.device)
                images, labels = images.to(self.device), labels.to(self.device)

                batch_size = len(images)

                logits = self.model(images)

                outputs = F.softmax(logits, dim=1)
                _, predicted = torch.max(outputs.data, -1)

                collection_Ms = []
                expert_predictions = []
                for expert in self.experts_list:
                    exp_preds = torch.tensor(expert(data, labels, labels_sparse),
                                             device=self.device)
                    m = (exp_preds == labels).int()
                    collection_Ms.append(m)
                    expert_predictions.append(exp_preds)

                loss = self.pre_loss_fn(logits, labels, collection_Ms, self.pre_outdim, with_softmax=True)
                losses.append(loss.item())

                for i in range(0, batch_size):
                    r = (predicted[i].item() >= self.pre_classes)
                    if predicted[i] >= self.pre_classes:
                        max_idx = 0
                        # get second max
                        for j in range(0, self.pre_classes):
                            if outputs.data[i][j] >= outputs.data[i][max_idx]:
                                max_idx = j
                        prediction = max_idx
                    else:
                        prediction = predicted[i]
                    alone_correct += (prediction == labels[i]).item()
                    if r == 0:
                        total += 1
                        correct += (predicted[i] == labels[i]).item()
                        correct_sys += (predicted[i] == labels[i]).item()
                    if r == 1:
                        deferred_exp = (predicted[i] - self.pre_classes).item()
                        exp_prediction = expert_predictions[deferred_exp][i]
                        exp += (exp_prediction == labels[i]).item()
                        exp_total += 1
                        expert_correct_dic[deferred_exp] += (
                                exp_prediction == labels[i]).item()
                        expert_total_dic[deferred_exp] += 1
                        correct_sys += (exp_prediction == labels[i]).item()

                    real_total += 1

        cov = str(total) + str(" out of") + str(real_total)

        #  === Individual Expert Accuracies === #
        expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002)
                             for k
                             in range(len(self.experts_list))}
        # Add expert accuracies dict
        to_print = {"coverage": cov, "system_accuracy": 100 * correct_sys / real_total,
                    "expert_accuracy": 100 * exp / (exp_total + 0.0002),
                    "classifier_accuracy": 100 * correct / (total + 0.0001),
                    "alone_classifier": 100 * alone_correct / real_total,
                    "validation_loss": np.average(losses),
                    "n_experts": len(self.experts_list),
                    **expert_accuracies}
        print(to_print, flush=True)
        return to_print


    def validate(self):
        correct = 0
        correct_sys = 0
        exp = 0
        exp_total = 0
        total = 0
        real_total = 0
        alone_correct = 0
        #  === Individual Expert Accuracies === #
        expert_correct_dic = {k: 0 for k in range(self.meta_experts)}
        expert_total_dic = {k: 0 for k in range(self.meta_experts)}
        losses = []
        for i, batch in enumerate(self.meta_valid_loader, 1):
            if len(batch) == 2:
                data, label = [_.to(self.device) for _ in batch]
                labels_sparse = None
            else:
                data, label, labels_sparse = [_.to(self.device) for _ in batch]
            # sample expert predictions for context
            experts_sample = np.random.choice(self.experts_train, self.meta_experts).tolist()
            expert_cntx = self.cntx_sampler_val.sample(n_experts=1)
            images_cntx = expert_cntx.xc[0]
            targets_cntx = expert_cntx.yc[0]
            cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse[0]
            collection_Ms = []
            for expert in experts_sample:
                exp_preds_cntx = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], cntx_yc_sparse),
                                              device=self.device)
                m = (exp_preds_cntx == targets_cntx).int()
                collection_Ms.append(m)

            logits_q = self.model((images_cntx, targets_cntx, data, collection_Ms))
            collection_Ms = []
            expert_predictions = []
            for expert in experts_sample:
                exp_preds = torch.tensor(expert(data, label, labels_sparse), device=self.device)
                m = (exp_preds == label).int()
                collection_Ms.append(m)
                expert_predictions.append(exp_preds)

            loss_q = self.meta_loss_fn(logits_q, label, collection_Ms, self.meta_experts + self.meta_classes, with_softmax=True)
            batch_size = len(data)
            outputs = F.softmax(logits_q, dim=1)
            _, predicted = torch.max(outputs.data, -1)
            losses.append(loss_q.item())
            for i in range(0, batch_size):
                r = (predicted[i].item() >= self.meta_classes)
                if predicted[i] >= self.meta_classes:
                    max_idx = 0
                    # get second max
                    for j in range(0, self.meta_classes):
                        if outputs.data[i][j] >= outputs.data[i][max_idx]:
                            max_idx = j
                    prediction = max_idx
                else:
                    prediction = predicted[i]
                alone_correct += (prediction == label[i]).item()
                if r == 0:
                    total += 1
                    correct += (predicted[i] == label[i]).item()
                    correct_sys += (predicted[i] == label[i]).item()
                if r == 1:
                    deferred_exp = (predicted[i] - self.pre_classes).item()
                    exp_prediction = expert_predictions[deferred_exp][i]
                    exp += (exp_prediction == label[i]).item()
                    exp_total += 1
                    expert_correct_dic[deferred_exp] += (
                            exp_prediction == label[i]).item()
                    expert_total_dic[deferred_exp] += 1
                    correct_sys += (exp_prediction == label[i]).item()

                real_total += 1
        print(losses)
        cov = str(total) + str(" out of") + str(real_total)

        #  === Individual Expert Accuracies === #
        expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002)
                             for k
                             in range(self.meta_experts)}
        # Add expert accuracies dict
        to_print = {"coverage": cov, "system_accuracy": 100 * correct_sys / real_total,
                    "expert_accuracy": 100 * exp / (exp_total + 0.0002),
                    "classifier_accuracy": 100 * correct / (total + 0.0001),
                    "alone_classifier": 100 * alone_correct / real_total,
                    "validation_loss": np.average(losses),
                    "n_experts": self.meta_experts,
                    **expert_accuracies}
        print(to_print, flush=True)
        return to_print