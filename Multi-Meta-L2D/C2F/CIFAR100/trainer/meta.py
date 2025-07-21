""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import math
import time

from CIFAR100.dataloader.samplers import CategoriesSampler
from CIFAR100.models.mtl import MtlLearner
from CIFAR100.utils.misc import *
from CIFAR100.dataloader.dataset_loader import DatasetLoader as Dataset
from CIFAR100.dataloader.dataset_loader import CIFAR10_Sampled
from CIFAR100.lib.losses import multi_reject_CrossEntropyLoss, reject_CrossEntropyLoss
from CIFAR100.lib.experts import Cifar20SyntheticExpert
from CIFAR100.lib.dataset import load_cifar, ContextSampler
from CIFAR100.utils.misc import get_logger, accuracy


class MetaTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)

        save_path1 = '_'.join([args.dataset, args.model_type, 'MTL'])
        save_path2 = 'step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr1' + str(
            args.meta_lr1) + '_lr2' + str(args.meta_lr2) + \
                     '_maxepoch' + str(args.max_epoch) + '_baselr' + str(args.base_lr) + \
                     '_updatestep' + str(args.update_step) + '_stepsize' + str(
            args.step_size) + '_pre' + args.name + 'meta' + args.meta_label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        self.train_logger = get_logger(os.path.join(args.save_path, "meta_train.log"))
        self.test_logger = get_logger(os.path.join(args.save_path, "meta_test.log"))


        self.args = args
        self.device = args.device
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

        train_data, val_data, test_data = load_cifar(variety='20_100', data_aug=True, seed=args.seed)
        kwargs = {'num_workers': 4, 'pin_memory': True}
        self.cntx_sampler_train = ContextSampler(train_data.data, train_data.targets, train_data.transform,
                                                 train_data.targets_sparse, \
                                                 n_cntx_pts=args.n_cntx_pts, device=self.device, **kwargs)

        # Context sampler val/test-time: partition val/test sets
        prop_cntx = 0.2
        val_cntx_size = int(prop_cntx * len(val_data))
        val_data_cntx, val_data_trgt = torch.utils.data.random_split(val_data,
                                                                     [val_cntx_size, len(val_data) - val_cntx_size], \
                                                                     generator=torch.Generator().manual_seed(
                                                                         args.seed))
        test_cntx_size = int(prop_cntx * len(test_data))
        test_data_cntx, test_data_trgt = torch.utils.data.random_split(test_data, [test_cntx_size,
                                                                                   len(test_data) - test_cntx_size], \
                                                                       generator=torch.Generator().manual_seed(
                                                                           args.seed))
        self.cntx_sampler_val = ContextSampler(images=val_data_cntx.dataset.data[val_data_cntx.indices],
                                               labels=val_data_cntx.dataset.targets[val_data_cntx.indices],
                                               transform=val_data.transform,
                                               labels_sparse=val_data_cntx.dataset.targets_sparse[
                                                   val_data_cntx.indices] if
                                               args.dataset == 'cifar20_100' else None,
                                               n_cntx_pts=args.n_cntx_pts, device=self.device, **kwargs)
        self.cntx_sampler_test = ContextSampler(images=test_data_cntx.dataset.data[test_data_cntx.indices],
                                                labels=np.array(test_data_cntx.dataset.targets)[test_data_cntx.indices],
                                                transform=test_data.transform,
                                                labels_sparse=test_data_cntx.dataset.targets_sparse[
                                                    test_data_cntx.indices] if
                                                args.dataset == 'cifar20_100' else None,
                                                n_cntx_pts=args.n_cntx_pts, device=self.device, **kwargs)

        self.train_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=args.meta_batch_size, shuffle=True,
                                                        **kwargs)  # drop_last=True
        self.valid_loader = torch.utils.data.DataLoader(val_data_trgt,
                                                        batch_size=args.meta_val_batch_size, shuffle=False,
                                                        **kwargs)  # shuffle=True, drop_last=True
        self.test_loader = torch.utils.data.DataLoader(test_data_trgt, batch_size=args.meta_test_batch_size,
                                                       shuffle=False,
                                                       **kwargs)

        self.meta_loss_fn = multi_reject_CrossEntropyLoss


        self.model = MtlLearner(self.args, self.meta_loss_fn)

        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
             {'params': self.model.base_learner.parameters(), 'lr': self.args.meta_lr2}], lr=self.args.meta_lr1)
        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                            gamma=self.args.gamma)

        # load pretrained model without FC classifier
        self.model_dict = self.model.state_dict()
        if self.args.init_weights is not None:
            pretrained_dict = torch.load(self.args.init_weights)['params']
        else:
            pre_base_dir = osp.join(log_base_dir, 'pre')
            pre_save_path1 = '_'.join([args.dataset, args.model_type])
            pre_save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(
                args.pre_gamma) + '_step' + \
                             str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch) + '_exp' + str(args.name)
            pre_save_path = pre_base_dir + '/' + pre_save_path1 + '_' + pre_save_path2
            pretrained_dict = torch.load(osp.join(pre_save_path, 'max_acc.pth'))['params']
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
        self.train_logger.info(pretrained_dict.keys())
        self.model_dict.update(pretrained_dict)
        self.model.load_state_dict(self.model_dict)
        param_num = sum(map(lambda x: np.prod(x.shape), self.model.parameters()))
        self.train_logger.info("Total param {0}".format(param_num))

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.to(self.device)

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        """The function for the meta-train phase."""

        # Set the meta-train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Start meta-train
        for epoch in range(1, self.args.max_epoch + 1):
            # # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number
                global_count = global_count + 1

                if len(batch) == 2:
                    data, label = [_.to(self.device) for _ in batch]
                    labels_sparse = None
                else:
                    data, label, labels_sparse = [_.to(self.device) for _ in batch]

                # sample expert predictions for context
                experts_sample = np.random.choice(self.experts_train, self.meta_experts).tolist()
                expert_cntx = self.cntx_sampler_train.sample(n_experts=1)
                images_cntx = expert_cntx.xc.squeeze(0)
                label_cntx = expert_cntx.yc.squeeze(0)
                cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse.squeeze(0)
                collection_Ms = []
                for expert in experts_sample:
                    exp_preds_cntx = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], cntx_yc_sparse),
                                                  device=self.device)
                    m = (exp_preds_cntx == label_cntx).int()
                    collection_Ms.append(m)

                logits_q = self.model((images_cntx, label_cntx, data, collection_Ms))

                collection_Ms = []
                expert_predictions = []
                for expert in experts_sample:
                    exp_preds = torch.tensor(expert(data, label, labels_sparse), device=self.device)
                    m = (exp_preds == label).int()
                    collection_Ms.append(m)
                    expert_predictions.append(exp_preds)

                loss = self.meta_loss_fn(logits_q, label, collection_Ms, self.meta_experts + self.meta_classes,
                                         with_softmax=True)

                outputs = F.softmax(logits_q, dim=-1)
                prec1 = accuracy(outputs.data[:, :self.meta_classes], label, topk=(1,))[0].item()
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss.item()), global_count)
                writer.add_scalar('data/acc', float(prec1), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), prec1))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(prec1)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.train_logger.info('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), prec1))

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # Start validation for this epoch, set model to eval mode
            self.model.eval()

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Print previous information
            if epoch % 10 == 0:
                self.train_logger.info(
                    'Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))


            metrics = self.evaluate(self.valid_loader, self.cntx_sampler_val, self.experts_train)
            test = self.evaluate(self.test_loader, self.cntx_sampler_test, self.experts_test)
            valid_loss = metrics['val_loss']
            valid_acc = metrics['sys_acc']
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(valid_loss), epoch)
            writer.add_scalar('data/val_acc', float(valid_acc), epoch)
            # Print loss and accuracy for this epoch
            self.train_logger.info('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, valid_loss, valid_acc))

            # Update best saved model
            if valid_acc > trlog['max_acc']:
                trlog['max_acc'] = valid_acc
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch' + str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(valid_loss)
            trlog['val_acc'].append(valid_acc)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                self.train_logger.info('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(
                    epoch / self.args.max_epoch)))

        writer.close()

    def evaluate(self, data_loader, cntx_sampler, expert_list, logger=None):

        correct = 0
        correct_sys = 0
        exp = 0
        exp_total = 0
        total = 0
        real_total = 0
        clf_alone_correct = 0
        exp_alone_correct = 0
        #  === Individual Expert Accuracies === #
        expert_correct_dic = {k: 0 for k in range(self.meta_experts)}
        expert_total_dic = {k: 0 for k in range(self.meta_experts)}
        #  === Individual Expert Accuracies === #

        losses = []
        confidence_diff = []
        is_rejection = []
        clf_predictions = []
        exp_predictions = [[] for i in range(self.meta_experts)]
        defer_exps = []
        for data in data_loader:
            if len(data) == 2:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                labels_sparse = None
            else:
                images, labels, labels_sparse = data
                images, labels, labels_sparse = images.to(self.device), labels.to(self.device), labels_sparse.to(
                    self.device)

            experts_sample = np.random.choice(expert_list, self.meta_experts).tolist()

            # sample expert predictions for context
            expert_cntx = cntx_sampler.sample(n_experts=1)
            images_cntx = expert_cntx.xc.squeeze(0)
            targets_cntx = expert_cntx.yc.squeeze(0)
            cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse.squeeze(0)

            collection_Ms = []
            for expert in experts_sample:
                exp_preds_cntx = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], cntx_yc_sparse),
                                              device=self.device)
                costs = (exp_preds_cntx == expert_cntx.yc.squeeze(0)).int()
                collection_Ms.append(costs)

            logits_q = self.model((images_cntx, targets_cntx, images, collection_Ms))

            with torch.no_grad():
                if self.args.loss_type == "ova":
                    probs = F.sigmoid(logits_q)
                else:
                    probs = F.softmax(logits_q, dim=-1)

                clf_probs, clf_preds = probs[:, :self.meta_classes].max(dim=-1)
                exp_probs, defer_exp = probs[:, self.meta_classes:].max(dim=-1)
                defer_exps.append(defer_exp)
                confidence_diff.append(clf_probs - exp_probs)
                clf_predictions.append(clf_preds)
                # defer if rejector logit strictly larger than (max of) classifier logits
                # since max() returns index of first maximal value (different from paper (geq))
                _, predicted = probs.max(dim=-1)
                is_rejection.append((predicted >= self.meta_classes).int())
                # print("is rej",is_rejection)

                collection_Ms = []
                for idx, expert in enumerate(experts_sample):
                    exp_preds = torch.tensor(expert(images, labels, labels_sparse),
                                             device=self.device)
                    costs = (exp_preds == labels).int()
                    collection_Ms.append(costs)
                    exp_predictions[idx].append(exp_preds)

                loss = self.meta_loss_fn(logits_q, labels, collection_Ms, self.meta_classes + self.meta_experts,
                                         with_softmax=True)
                losses.append(loss.item())

        confidence_diff = torch.cat(confidence_diff)
        indices_order = confidence_diff.argsort()

        is_rejection = torch.cat(is_rejection)[indices_order]
        clf_predictions = torch.cat(clf_predictions)[indices_order]
        exp_predictions = [torch.cat(exp_predictions[i])[indices_order] for i in range(self.meta_experts)]
        defer_exps = torch.cat(defer_exps)[indices_order]

        kwargs = {'num_workers': 0, 'pin_memory': True}
        data_loader_new = torch.utils.data.DataLoader(
            torch.utils.data.Subset(data_loader.dataset, indices=indices_order),
            batch_size=data_loader.batch_size, shuffle=False, **kwargs)

        max_defer = math.floor(1.0 * len(data_loader.dataset))

        for data in data_loader_new:
            if len(data) == 2:
                images, labels = data
            else:
                images, labels, _ = data
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = len(images)

            for i in range(0, batch_size):
                defer_running = is_rejection[:real_total].sum().item()
                if defer_running >= max_defer:
                    r = 0
                else:
                    r = is_rejection[real_total].item()
                prediction = clf_predictions[real_total].item()
                defer_exp = defer_exps[real_total].item()
                exp_prediction = exp_predictions[defer_exp][real_total].item()

                clf_alone_correct += (prediction == labels[i]).item()
                exp_alone_correct += (exp_prediction == labels[i].item())
                if r == 0:
                    total += 1
                    correct += (prediction == labels[i]).item()
                    correct_sys += (prediction == labels[i]).item()
                if r == 1:
                    exp += (exp_prediction == labels[i].item())
                    correct_sys += (exp_prediction == labels[i].item())
                    exp_total += 1
                    # Individual Expert Accuracy ===
                    expert_correct_dic[defer_exp] += (exp_prediction == labels[i].item())
                    expert_total_dic[defer_exp] += 1

                real_total += 1
        #  === Individual Expert Accuracies === #
        expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002)
                             for k
                             in range(self.meta_experts)}
        cov = str(total) + str("/") + str(real_total)
        metrics = {"cov": cov, "sys_acc": 100 * correct_sys / real_total,
                   "exp_acc": 100 * exp / (exp_total + 0.0002),
                   "clf_acc": 100 * correct / (total + 0.0001),
                   "exp_acc_alone": 100 * exp_alone_correct / real_total,
                   "clf_acc_alone": 100 * clf_alone_correct / real_total,
                   "val_loss": np.average(losses),
                   **expert_accuracies}
        to_print = ""
        for k, v in metrics.items():
            if type(v) == str:
                to_print += f"{k} {v} "
            else:
                to_print += f"{k} {v:.6f} "
        if logger is not None:
            logger.info(to_print)
        else:
            print(to_print)
        return metrics

    def eval(self):
        """The function for the meta-eval phase."""
        # Load model for meta-test phase
        if self.args.eval_weights is not None:
            self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
        else:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params'])
        # Set model to eval mode
        self.model.eval()
        self.evaluate(self.test_loader, self.cntx_sampler_test, self.experts_test, self.test_logger)




