##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/Sha-Lab/FEAT
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for meta-train phase. """
from torch.utils.data import DataLoader
import math

# from CIFAR100.dataloader.samplers import CategoriesSampler
# from CIFAR100.models.mtl import MtlLearner
# from CIFAR100.utils.misc import *
# from CIFAR100.dataloader.dataset_loader import DatasetLoader as Dataset
# from CIFAR100.dataloader.dataset_loader import CIFAR10_Sampled
# from CIFAR100.lib.losses import multi_reject_CrossEntropyLoss, reject_CrossEntropyLoss
# from CIFAR100.lib.experts import Cifar20SyntheticExpert
# from CIFAR100.lib.dataset import load_cifar, ContextSampler
# from CIFAR100.utils.misc import get_logger, accuracy

import os.path as osp
import tqdm

import medmnist
from lib.experts import SyntheticExpertOverlap
from lib.losses import softmax
from lib.datasets import ContextSampler
from utils.misc import *
from models.mtl import MtlLearner
import numpy as np
import PIL
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import INFO
from tensorboardX import SummaryWriter


class MetaTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""

    def __init__(self, args):

        if args.phase == 'meta_train':
            time_stamp = time.strftime('%y%m%d_%H%M%S')
            args.time_stamp_meta = time_stamp
        else:
            time_stamp = args.time_stamp_meta
            if time_stamp is None:
                print("please assign the timestamp")
                return

        save_file = str(args.model_flag) + 'step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr1' + str(
            args.meta_lr1) + '_lr2' + str(args.meta_lr2) + \
                     '_epochs' + str(args.meta_epochs) + '_baselr' + str(args.base_lr) + \
                     '_updatestep' + str(args.update_step) + '_stepsize' + str(
            args.step_size)

        args.save_path = f"{args.log_dir}/pre/{args.dataset}/{args.loss_type}/p{str(args.p_out)}_seed{str(args.seed)}" \
                         f"/preExp{args.pre_experts}_metaExp{args.meta_experts}/" \
                         f"/depth{args.depth}_hid{args.hidden_dim}/{save_file}/{args.time_stamp_meta}"
        os.makedirs(args.save_path, exist_ok=True)


        self.train_logger = get_logger(os.path.join(args.save_path, "meta_train.log"))
        self.test_logger = get_logger(os.path.join(args.save_path, "meta_test.log"))

        str_ids = args.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        if len(gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')


        self.args = args
        info = INFO[args.dataset]
        self.task = info['task']
        as_rgb = args.as_rgb
        size = args.size
        args.n_channels = 3 if as_rgb else info['n_channels']
        n_classes = len(info['label'])
        self.num_classes = args.num_classes = n_classes
        self.meta_experts = args.meta_experts

        sample_expert = 10  # assume exactly divisible by 2
        pop_dict = {"pop1": [i for i in range(0, 5)],
                    "pop2": [i for i in range(5, 11)]}
        self.experts_test, self.experts_train = [], []
        can_class_train, can_class_test = [], []

        for pop_k, pop_v in pop_dict.items():
            train_exp = []
            for _ in range(sample_expert):  # train
                class_num = np.random.randint(1, len(pop_v) + 1)
                class_oracle = np.random.choice(pop_v, class_num, replace=False)
                expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=args.num_classes, p_in=1.0,
                                                p_out=args.p_out)
                train_exp.append(expert)
                can_class_train.append(class_oracle.tolist())
                train_exp.append(expert)
            self.experts_train += train_exp
            self.experts_test += train_exp[
                                 :sample_expert // 2]  # pick 50% experts from experts_train (order not matter)

        for pop_k, pop_v in pop_dict.items():
            for _ in range(sample_expert // 2):  # then sample 50% new experts
                class_num = np.random.randint(1, len(pop_v) + 1)
                class_oracle = np.random.choice(pop_v, class_num, replace=False)
                expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=args.num_classes, p_in=1.0,
                                                p_out=args.p_out)
                can_class_test.append(class_oracle)
                self.experts_test.append(expert)
        print("Train Experts => ", can_class_train)
        print("Test Experts => ", can_class_test)

        DataClass = getattr(medmnist, info['python_class'])
        TestDataClass = getattr(medmnist, INFO[args.test_dataset]['python_class'])


        print('==> Preparing data...')

        if args.resize:
            data_transform = transforms.Compose(
                [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[.5], std=[.5])])
        else:
            data_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean=[.5], std=[.5])])

        train_data = DataClass(split='train', transform=data_transform, download=args.download, as_rgb=as_rgb,
                               size=size)
        val_data = DataClass(split='val', transform=data_transform, download=args.download, as_rgb=as_rgb, size=size)
        test_data = TestDataClass(split='test', transform=data_transform, download=args.download, as_rgb=as_rgb, size=size)

        kwargs = {'num_workers': 4, 'pin_memory': True}

        self.cntx_sampler_train = ContextSampler(train_data.imgs, train_data.labels, train_data.transform,
                                                 None, n_cntx_pts=args.n_cntx_pts, device=self.device, **kwargs)

        prop_cntx = 0.2
        val_cntx_size = int(prop_cntx * len(val_data))
        val_data_cntx, val_data_trgt = torch.utils.data.random_split(val_data,
                                                                     [val_cntx_size, len(val_data) - val_cntx_size], \
                                                                     generator=torch.Generator().manual_seed(
                                                                         args.seed))

        test_cntx_size = int(prop_cntx * len(test_data))
        test_data_cntx, test_data_trgt = torch.utils.data.random_split(test_data,
                                                                       [test_cntx_size,
                                                                        len(test_data) - test_cntx_size], \
                                                                       generator=torch.Generator().manual_seed(
                                                                           args.seed))
        self.cntx_sampler_val = ContextSampler(images=val_data_cntx.dataset.imgs[val_data_cntx.indices],
                                               labels=val_data_cntx.dataset.labels[val_data_cntx.indices],
                                               transform=val_data.transform,
                                               labels_sparse=None,
                                               n_cntx_pts=args.n_cntx_pts, device=self.device, **kwargs)
        self.cntx_sampler_test = ContextSampler(images=test_data_cntx.dataset.imgs[test_data_cntx.indices],
                                                labels=np.array(test_data_cntx.dataset.labels)[test_data_cntx.indices],
                                                transform=test_data.transform,
                                                labels_sparse=None,
                                                n_cntx_pts=args.n_cntx_pts, device=self.device, **kwargs)

        self.train_loader = data.DataLoader(dataset=train_data,batch_size=args.meta_batch_size,
                                       shuffle=True, **kwargs)
        self.valid_loader = data.DataLoader(dataset=val_data_trgt,batch_size=args.meta_val_batch_size,
                                     shuffle=False, **kwargs)
        self.test_loader = data.DataLoader(dataset=test_data_trgt,batch_size=args.meta_test_batch_size,
                                      shuffle=False, **kwargs)


        self.meta_loss_fn = softmax


        self.model = MtlLearner(self.args, self.meta_loss_fn, mode='meta')

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
            save_file = str(args.model_flag) + '_bz' + str(args.pre_batch_size) + '_lr' + str(
                args.pre_lr) + '_gamma' + str(
                args.pre_gamma) + '_step' + str(args.pre_step_size) + '_epochs' + str(args.pre_epochs)

            pre_save_path = f"{args.log_dir}/pre/{args.dataset}/{args.loss_type}/" \
                             f"p{str(args.p_out)}_seed{str(args.seed)}" \
                             f"/preExp{args.pre_experts}_metaExp{args.meta_experts}/{save_file}/{args.time_stamp_pre}"

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

        self.train_logger.info(self.args)

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Start meta-train
        for epoch in range(1, self.args.meta_epochs + 1):
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

                input, label = batch
                input = input.to(self.device)
                labels_sparse = None

                if self.task == 'multi-label, binary-class':
                    label = label.to(torch.float32).to(self.device)
                else:
                    label = torch.squeeze(label, 1).long().to(self.device)


                # sample expert predictions for context
                experts_sample = np.random.choice(self.experts_train, self.meta_experts).tolist()
                expert_cntx = self.cntx_sampler_train.sample(n_experts=1)
                cntx_yc_sparse = None
                collection_Ms = []
                for expert in experts_sample:
                    exp_preds_cntx = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], cntx_yc_sparse),
                                                  device=self.device)
                    m = (exp_preds_cntx == expert_cntx.yc[0]).int()
                    collection_Ms.append(m)

                logits_q = self.model((expert_cntx.xc[0], expert_cntx.yc[0], input, collection_Ms))

                collection_Ms = []
                expert_predictions = []
                for expert in experts_sample:
                    exp_preds = torch.tensor(expert(input, label, labels_sparse), device=self.device)
                    m = (exp_preds == label).int()
                    collection_Ms.append(m)
                    expert_predictions.append(exp_preds)

                loss = self.meta_loss_fn(logits_q, label, collection_Ms, self.meta_experts + self.num_classes)

                outputs = F.softmax(logits_q, dim=-1)
                prec1 = accuracy(outputs.data[:, :self.num_classes], label, topk=(1,))[0].item()
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

            # | ===============================================================================
            # | Start Meta validation phase
            metrics = self.evaluate(self.valid_loader, self.cntx_sampler_val, self.experts_train)
            print("测试")
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
                    epoch / self.args.meta_epochs)))

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

            images, labels = data
            images = images.to(self.device)
            labels_sparse = None

            if self.task == 'multi-label, binary-class':
                labels = labels.to(torch.float32).to(self.device)
            else:
                labels = torch.squeeze(labels, 1).long().to(self.device)

            experts_sample = np.random.choice(expert_list, self.meta_experts).tolist()

            # sample expert predictions for context
            expert_cntx = cntx_sampler.sample(n_experts=1)
            cntx_yc_sparse = None

            collection_Ms = []
            for expert in experts_sample:
                exp_preds_cntx = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], cntx_yc_sparse),
                                              device=self.device)
                costs = (exp_preds_cntx == expert_cntx.yc[0]).int()
                collection_Ms.append(costs)

            logits_q = self.model((expert_cntx.xc[0], expert_cntx.yc[0], images, collection_Ms))

            with torch.no_grad():
                if self.args.loss_type == "ova":
                    probs = F.sigmoid(logits_q)
                else:
                    probs = F.softmax(logits_q, dim=-1)

                clf_probs, clf_preds = probs[:, :self.num_classes].max(dim=-1)
                exp_probs, defer_exp = probs[:, self.num_classes:].max(dim=-1)
                defer_exps.append(defer_exp)
                confidence_diff.append(clf_probs - exp_probs)
                clf_predictions.append(clf_preds)
                # defer if rejector logit strictly larger than (max of) classifier logits
                # since max() returns index of first maximal value (different from paper (geq))
                _, predicted = probs.max(dim=-1)
                is_rejection.append((predicted >= self.num_classes).int())
                # print("is rej",is_rejection)

                collection_Ms = []
                for idx, expert in enumerate(experts_sample):
                    exp_preds = torch.tensor(expert(images, labels, labels_sparse),
                                             device=self.device)
                    costs = (exp_preds == labels).int()
                    collection_Ms.append(costs)
                    exp_predictions[idx].append(exp_preds)

                loss = self.meta_loss_fn(logits_q, labels, collection_Ms, self.num_classes + self.meta_experts)
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
        self.test_logger.info(self.args)
        """The function for the meta-eval phase."""
        # Load model for meta-test phase
        if self.args.eval_weights is not None:
            self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
        else:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params'])
        # Set model to eval mode
        self.model.eval()
        self.evaluate(self.test_loader, self.cntx_sampler_test, self.experts_test, self.test_logger)




