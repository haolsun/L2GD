""" Trainer for pretrain phase. """
import os.path as osp
from torch.utils.data import DataLoader

import medmnist
from lib.experts import SyntheticExpertOverlap
from lib.losses import softmax, ova
from lib.datasets import ContextSampler
from utils.misc import *
from models.mtl import MtlLearner
import numpy as np
import PIL
import torchvision.transforms as transforms
from medmnist import INFO


class PreTrainer(object):
    """The class that contains the code for the pretrain phase."""

    def __init__(self, args):

        self.args = args
        self.pre_experts = args.pre_experts
        self.meta_experts = args.meta_experts


        save_file = str(args.model_flag) + '_bz' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(
            args.pre_gamma) + '_step' + str(args.pre_step_size) + '_epochs' + str(args.pre_epochs)

        args.save_path = f"{args.log_dir}/pre/{args.dataset}/{args.loss_type}/" \
                            f"p{str(args.p_out)}_seed{str(args.seed)}" \
                            f"/preExp{args.pre_experts}_metaExp{args.meta_experts}/{save_file}/{time.strftime('%y%m%d_%H%M%S')}"
        os.makedirs(args.save_path, exist_ok=True)

        self.logger = get_logger(os.path.join(args.save_path, "pre_train.log"))


        str_ids = args.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        if len(gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

        info = INFO[args.dataset]
        self.task = info['task']
        as_rgb = args.as_rgb
        size = args.size
        args.n_channels = 3 if as_rgb else info['n_channels']
        n_classes = len(info['label'])
        self.num_classes = args.num_classes = n_classes


        DataClass = getattr(medmnist, info['python_class'])

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

        train_data = DataClass(split='train', transform=data_transform, download=args.download, as_rgb=as_rgb, size=size)
        val_data = DataClass(split='val', transform=data_transform, download=args.download, as_rgb=as_rgb, size=size)

        kwargs = {'num_workers': 4, 'pin_memory': True}

        self.train_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=args.pre_batch_size, shuffle=True,
                                                        **kwargs)
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
        self.cntx_sampler_val = ContextSampler(images=val_data_cntx.dataset.imgs[val_data_cntx.indices],
                                               labels=val_data_cntx.dataset.labels[val_data_cntx.indices],
                                               transform=val_data.transform,
                                               labels_sparse=None,
                                               n_cntx_pts=args.n_cntx_pts, device=self.device, **kwargs)
        self.meta_valid_loader = torch.utils.data.DataLoader(val_data_trgt,
                                                             batch_size=64, shuffle=False,
                                                             **kwargs)



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
        self.experts_list = np.random.choice(self.experts_train, self.pre_experts).tolist()
        print("Train Experts => ", can_class_train)
        print("Test Experts => ", can_class_test)



        # | ====================================================
        # | Set Loss function
        if args.loss_type == 'softmax':
            self.loss_fn = softmax
        else:  # ova
            self.loss_fn = ova


        # | ====================================================
        # | Build pretrain model
        self.pre_outdim = self.num_classes + self.pre_experts
        self.model = MtlLearner(self.args, self.loss_fn, mode='pre')
        param_num = sum(map(lambda x: np.prod(x.shape), self.model.parameters()))
        self.logger.info("Total param" + str(param_num))


        self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.pre_lr}, \
                                          {'params': self.model.pre_fc.parameters(), 'lr': self.args.pre_lr}], \
                                         momentum=self.args.pre_custom_momentum, nesterov=True,
                                         weight_decay=self.args.pre_custom_weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, len(self.train_loader) * args.pre_epochs)


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
        self.logger.info(self.args)

        # | ===============================================================================
        # | Start Pretrain phase
        best_validation_loss = np.inf
        for epoch in range(1, self.args.pre_epochs + 1):

            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()

            self.model.train()
            self.model.mode = 'pre'

            end = time.time()
            epoch_train_loss = []

            # Using tqdm to read samples from train loader
            for i, batch in enumerate(self.train_loader):

                input, target = batch
                input = input.to(self.device)
                target_sparse = None

                if self.task == 'multi-label, binary-class':
                    target = target.to(torch.float32).to(self.device)
                else:
                    target = torch.squeeze(target, 1).long().to(self.device)


                # Output logits for model
                logits = self.model(input)
                # Calculate train loss
                collection_Ms = []
                expert_predictions = []
                for expert in self.experts_list:
                    exp_preds = torch.tensor(expert(input, target, target_sparse),
                                             device=self.device)
                    m = (exp_preds == target).int()
                    collection_Ms.append(m)
                    expert_predictions.append(exp_preds)

                # 2.损失函数是否有误
                loss = self.loss_fn(logits, target, collection_Ms, self.pre_outdim)
                outputs = F.softmax(logits, dim=1)
                epoch_train_loss.append(loss.item())

                # measure accuracy and record loss
                prec1 = accuracy(outputs[:, :self.num_classes].data, target, topk=(1,))[0]
                losses.update(loss.data.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 1.放的位置是否对性能有影响
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

            # | ===============================================================================
            # | Start Validation for this epoch
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
        all_classes = self.num_classes + len(self.experts_list)

        losses = []
        self.model.eval()  # Crucial for networks with batchnorm layers!
        with torch.no_grad():
            for data in data_loader:

                images, labels = data
                images = images.to(self.device)
                labels_sparse = None

                if self.task == 'multi-label, binary-class':
                    labels = labels.to(torch.float32).to(self.device)
                else:
                    labels = torch.squeeze(labels, 1).long().to(self.device)

                batch_size = len(images)

                logits = self.model(images)

                # 系统的预测
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

                loss = self.loss_fn(logits, labels, collection_Ms, self.pre_outdim)
                losses.append(loss.item())

                for i in range(0, batch_size):
                    r = (predicted[i].item() >= self.num_classes)
                    if predicted[i] >= self.num_classes:
                        max_idx = 0
                        # get second max
                        for j in range(0, self.num_classes):
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
                        deferred_exp = (predicted[i] - self.num_classes).item()
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

            input, label = batch
            input = input.to(self.device)
            labels_sparse = None

            if self.task == 'multi-label, binary-class':
                label = label.to(torch.float32).to(self.device)
            else:
                label = torch.squeeze(label, 1).long().to(self.device)

            # sample expert predictions for context
            experts_sample = np.random.choice(self.experts_train, self.meta_experts).tolist()
            expert_cntx = self.cntx_sampler_val.sample(n_experts=1)
            collection_Ms = []
            for expert in experts_sample:
                exp_preds_cntx = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], None),
                                              device=self.device)
                m = (exp_preds_cntx == expert_cntx.yc[0]).int()
                collection_Ms.append(m)
            # 微调分类头
            logits_q = self.model((expert_cntx.xc[0], expert_cntx.yc[0], input, collection_Ms))
            # 计算
            collection_Ms = []
            expert_predictions = []
            for expert in experts_sample:
                exp_preds = torch.tensor(expert(input, label, labels_sparse), device=self.device)
                m = (exp_preds == label).int()
                collection_Ms.append(m)
                expert_predictions.append(exp_preds)

            loss_q = self.loss_fn(logits_q, label, collection_Ms, self.meta_experts + self.num_classes)
            batch_size = len(input)
            # 系统的预测
            outputs = F.softmax(logits_q, dim=1)
            _, predicted = torch.max(outputs.data, -1)
            losses.append(loss_q.item())
            for i in range(0, batch_size):
                r = (predicted[i].item() >= self.num_classes)
                if predicted[i] >= self.num_classes:
                    max_idx = 0
                    # get second max
                    for j in range(0, self.num_classes):
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
                    deferred_exp = (predicted[i] - self.num_classes).item()
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