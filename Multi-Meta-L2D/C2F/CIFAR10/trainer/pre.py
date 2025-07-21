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

from CIFAR10.dataloader.samplers import CategoriesSampler
from CIFAR10.models.mtl import MtlLearner
from CIFAR10.utils.misc import Averager, Timer, ensure_path
from CIFAR10.utils.misc import count_acc_sys_multiexp, count_acc_sys, count_acc, count_acc_l2d
from CIFAR10.dataloader.dataset_loader import DatasetLoader as Dataset
from CIFAR10.dataloader.dataset_loader import CIFAR10_Sampled
from CIFAR10.lib.dataset import load_cifar, load_ham10000, load_gtsrb, ContextSampler
from CIFAR10.lib.experts import SyntheticExpertOverlap, synth_expert2
from CIFAR10.lib.losses import reject_CrossEntropyLoss, multi_reject_CrossEntropyLoss
from CIFAR10.utils.misc import get_logger
from CIFAR10.utils.misc import multi_exp_evaluate, evaluate, evaluate_data


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
                     str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch) +'_exp'+str(args.name)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        self.logger = get_logger(os.path.join(args.save_path, "pre_train.log"))

        self.args = args
        self.device = args.device
        self.pre_classes = args.pre_classes
        self.pre_experts = args.pre_experts
        self.meta_classes = args.meta_classes
        self.meta_experts = args.meta_experts

        sample_expert = 6  # assume exactly divisible by 2
        pop_dict = {"pop1": [0, 1, 2],
                    "pop2": [3, 4, 5],
                    "pop3": [6, 7, 8, 9]}
        self.experts_list = []
        self.experts_test = []
        self.experts_train = []
        can_class_train = []
        can_class_test = []
        p_in = 1.0
        for pop_k, pop_v in pop_dict.items():
            train_exp = []
            for _ in range(sample_expert):  # train
                can_class_num = np.random.randint(1, len(pop_v) + 1)
                can_class_list = np.random.choice(pop_v, can_class_num, replace=False)
                can_class_train.append(can_class_list)
                expert = SyntheticExpertOverlap(classes_oracle=can_class_list, n_classes=self.meta_classes, p_in=p_in,
                                                p_out=0.1)
                train_exp.append(expert)
            self.experts_train += train_exp
            self.experts_test += train_exp[
                                 :sample_expert // 2]  # pick 50% experts from experts_train (order not matter)

        for pop_k, pop_v in pop_dict.items():
            for _ in range(sample_expert // 2):  # then sample 50% new experts
                can_class_num = np.random.randint(1, len(pop_v) + 1)
                can_class_list = np.random.choice(pop_v, can_class_num, replace=False)
                can_class_test.append(can_class_list)
                expert = SyntheticExpertOverlap(classes_oracle=can_class_list, n_classes=self.meta_classes, p_in=p_in,
                                                p_out=0.1)
                self.experts_test.append(expert)
        self.experts_list = np.random.choice(self.experts_train, self.pre_experts).tolist()

                
        train_data, val_data, test_data = load_cifar(variety='10', data_aug=False, seed=args.seed)

        kwargs = {'num_workers': 0, 'pin_memory': True}

        self.train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=args.pre_batch_size, shuffle=True,drop_last=True,
                                                   **kwargs)  # drop_last=True

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
                                                        batch_size=args.meta_val_batch_size, shuffle=False,
                                                        **kwargs)  # shuffle=True, drop_last=True


        self.pre_loss_fn = multi_reject_CrossEntropyLoss
        self.meta_loss_fn = multi_reject_CrossEntropyLoss

        self.pre_outdim = self.pre_classes + self.pre_experts
        self.model = MtlLearner(self.args, self.meta_loss_fn, mode='pre', num_cls=self.pre_outdim)
        param_num = sum(map(lambda x: np.prod(x.shape), self.model.parameters()))
        self.logger.info("Total param"+str(param_num))


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

        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            self.model.mode = 'pre'
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

                loss = self.pre_loss_fn(logits, label, collection_Ms, self.pre_outdim)

                # Calculate train accuracy
                acc = count_acc_sys_multiexp(logits, label, expert_predictions, self.pre_classes)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            self.logger.info('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()


            self.model.eval()
            self.model.mode = 'preval'

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Print previous information
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))


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
                loss_q = self.meta_loss_fn(logits_q, label, collection_Ms, self.meta_experts + self.meta_classes)
                metrics = evaluate_data(logits_q, expert_predictions, self.meta_classes, batch, self.device)
                val_loss_averager.add(loss_q.item())
                val_acc_averager.add(metrics['system_accuracy'])

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)
            # Print loss and accuracy for this epoch
            self.logger.info('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))

            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch' + str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                self.logger.info('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.args.max_epoch)))
        writer.close()