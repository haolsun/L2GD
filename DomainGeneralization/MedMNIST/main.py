import math
import random
import json
import copy
import itertools
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# local imports
from lib.utils import AverageMeter, accuracy, get_logger
from lib.losses import softmax, ova
from lib.experts import SyntheticExpertOverlap
from lib.modules import ClassifierRejector, ExpertWiseAggregator, ClassifierRejectorWithContextEmbedder
from lib.datasets import ContextSampler

import argparse
import os
import time

import medmnist
import numpy as np
import PIL
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import INFO
from lib.models import ResNet18, ResNet50
from torchvision.models import resnet18, resnet50


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate(model,
             experts_test,
             loss_fn,
             cntx_sampler,
             n_classes,
             data_loader,
             config,
             device,
             logger=None,
             budget=1.0,
             n_finetune_steps=0,
             lr_finetune=1e-1,
             p_cntx_inclusion=1.0):
    '''
    data loader : assumed to be instantiated with shuffle=False
    '''
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    clf_alone_correct = 0
    exp_alone_correct = 0
    #  === Individual Expert Accuracies === #
    expert_correct_dic = {k: 0 for k in range(config["train_experts"])}
    expert_total_dic = {k: 0 for k in range(config["train_experts"])}

    model.eval()  # Crucial for networks with batchnorm layers!
    if config["l2d"] == 'MAML':
        model.train()
    is_finetune = ((config["l2d"] == 'Multi') or (config["l2d"] == 'MAML')) and (n_finetune_steps > 0)
    if is_finetune:
        model_state_dict = model.state_dict()
        model_backup = copy.deepcopy(model)

    losses = []
    confidence_diff = []
    is_rejection = []
    clf_predictions = []
    exp_predictions = [[] for i in range(config["train_experts"])]
    defer_exps = []
    for data in data_loader:

        images, labels = data
        images = images.to(device)
        labels_sparse = None

        if config["task"] == 'multi-label, binary-class':
            labels = labels.to(torch.float32).to(device)
        else:
            labels = torch.squeeze(labels, 1).long().to(device)

        experts_sample = np.random.choice(
            experts_test, config["train_experts"]).tolist()

        if config['l2d'] == "EWA":
            # sample expert predictions for context
            expert_cntx = cntx_sampler.sample(n_experts=config["train_experts"])
            exp_preds_cntx = []
            for idx_exp, expert in enumerate(experts_sample):
                preds = torch.tensor(expert(expert_cntx.xc[idx_exp], expert_cntx.yc[idx_exp], None),
                                     device=device)
                exp_preds_cntx.append(preds.unsqueeze(0))
            expert_cntx.mc = torch.vstack(exp_preds_cntx)
        elif config['l2d'] == "pop":
            # sample expert predictions for context
            expert_cntx = cntx_sampler.sample(n_experts=1)
            exp_preds_cntx_list = []
            mc = []
            for expert in experts_sample:
                exp_preds_cntx = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], None),
                                              device=device)
                exp_preds_cntx_list.append(exp_preds_cntx.unsqueeze(0))
            mc.append(torch.vstack(exp_preds_cntx_list))
            expert_cntx.mc = mc
        else:
            # sample expert predictions for context
            expert_cntx = cntx_sampler.sample(n_experts=1)
            collection_Ms = []
            for expert in experts_sample:
                exp_preds_cntx = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], None),
                                              device=device)
                costs = (exp_preds_cntx == expert_cntx.yc.squeeze(0)).int()
                collection_Ms.append(costs)

        if is_finetune:
            model.train()

            # NB: could freeze base network like finetuning in train_epoch()
            for _ in range(n_finetune_steps):
                outputs_cntx = model(expert_cntx.xc[0])
                loss = loss_fn(outputs_cntx, expert_cntx.yc[0], collection_Ms, n_classes + config["train_experts"])
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for param in model.params.clf.parameters():
                        new_param = param - lr_finetune * param.grad
                        param.copy_(new_param)
            if config["l2d"] == 'Multi':
                model.eval()


        with torch.no_grad():
            # removes expert context based on coin flip
            coin_flip = np.random.binomial(1, p_cntx_inclusion)
            if coin_flip == 0:
                expert_cntx = None

            if config["l2d"] == 'EWA':
                outputs = model(images, expert_cntx)
            elif config["l2d"] == 'pop':
                outputs = model(images, expert_cntx).squeeze(0)
            else:
                outputs = model(images)

            if config["loss_type"] == "ova":
                probs = F.sigmoid(outputs)
            else:
                probs = outputs

            clf_probs, clf_preds = probs[:, :n_classes].max(dim=-1)
            exp_probs, defer_exp = probs[:, n_classes:].max(dim=-1)
            defer_exps.append(defer_exp)
            confidence_diff.append(clf_probs - exp_probs)
            clf_predictions.append(clf_preds)
            # defer if rejector logit strictly larger than (max of) classifier logits
            # since max() returns index of first maximal value (different from paper (geq))
            _, predicted = outputs.max(dim=-1)
            is_rejection.append((predicted >= n_classes).int())
            # print("is rej",is_rejection)

            collection_Ms = []
            for idx, expert in enumerate(experts_sample):
                exp_preds_cntx = torch.tensor(expert(images, labels, labels_sparse),
                                              device=device)
                costs = (exp_preds_cntx == labels).int()
                collection_Ms.append(costs)

                exp_predictions[idx].append(exp_preds_cntx)

            loss = loss_fn(outputs, labels, collection_Ms,
                           n_classes+config["train_experts"])
            losses.append(loss.item())

            if is_finetune:  # restore model on single-expert
                model = model_backup
                model.load_state_dict(copy.deepcopy(model_state_dict))
                if config["l2d"] == 'Multi':
                    model.eval()
                else:
                    model.train()


    confidence_diff = torch.cat(confidence_diff)
    indices_order = confidence_diff.argsort()

    is_rejection = torch.cat(is_rejection)[indices_order]
    clf_predictions = torch.cat(clf_predictions)[indices_order]
    exp_predictions = [torch.cat(exp_predictions[i])[indices_order]
                       for i in range(config["train_experts"])]
    defer_exps = torch.cat(defer_exps)[indices_order]

    kwargs = {'num_workers': 0, 'pin_memory': True}
    data_loader_new = torch.utils.data.DataLoader(torch.utils.data.Subset(data_loader.dataset, indices=indices_order),
                                                  batch_size=data_loader.batch_size, shuffle=False, **kwargs)

    max_defer = math.floor(budget * len(data_loader.dataset))

    for data in data_loader_new:
        if len(data) == 2:
            images, labels = data
        else:
            images, labels, _ = data
        images, labels = images.to(device), labels.to(device)
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
                expert_correct_dic[defer_exp] += (
                    exp_prediction == labels[i].item())
                expert_total_dic[defer_exp] += 1

            real_total += 1

    expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002)
                         for k
                         in range(config["train_experts"])}
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


def train_epoch(train_loader,
                model,
                optimizer_lst,
                scheduler_lst,
                epoch,
                experts_train,
                loss_fn,
                cntx_sampler,
                n_classes,
                config,
                logger,
                device,
                n_steps_maml=5,
                lr_maml=1e-1):
    """ Train for one epoch """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    epoch_train_loss = []

    for i, data in enumerate(train_loader):
        input, target = data
        input = input.to(device)
        target_sparse = None

        if config["task"] == 'multi-label, binary-class':
            target = target.to(torch.float32).to(device)
        else:
            target = torch.squeeze(target, 1).long().to(device)

        if (i == 0) and (config["l2d"] == 'MAML'):

            collection_Ms = []
            for expert in range(config["train_experts"]):
                costs = torch.zeros_like(target, device=target.device)
                collection_Ms.append(costs)

            outputs = model(input)
            loss = loss_fn(outputs, target, collection_Ms, n_classes + config["train_experts"])  # loss per expert

            loss.backward()
            model.zero_grad()


        if config["l2d"] == "EWA":
            expert_cntx = cntx_sampler.sample(
                n_experts=config["train_experts"])
            experts_sample = np.random.choice(
                experts_train, config["train_experts"]).tolist()
            exp_preds_cntx = []
            for idx_exp, expert in enumerate(experts_sample):
                cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse[idx_exp]
                preds = torch.tensor(expert(expert_cntx.xc[idx_exp], expert_cntx.yc[idx_exp], cntx_yc_sparse),
                                     device=device)
                exp_preds_cntx.append(preds.unsqueeze(0))
            expert_cntx.mc = torch.vstack(exp_preds_cntx)

            outputs = model(input, expert_cntx)  # [B,K+exp]

            collection_Ms = []
            for expert in experts_sample:
                exp_preds = torch.tensor(expert(input, target, target_sparse), device=device)
                costs = (exp_preds == target).int()
                collection_Ms.append(costs)
            loss = loss_fn(outputs, target, collection_Ms,
                           n_classes + config["train_experts"])


            prec1 = accuracy(outputs.data[:, :n_classes], target, topk=(1,))[0]
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            for optimizer in optimizer_lst:
                optimizer.zero_grad()
            loss.backward()
            for optimizer, scheduler in zip(optimizer_lst, scheduler_lst):
                optimizer.step()
                scheduler.step()

        elif config["l2d"] == "pop":
            expert_cntx = cntx_sampler.sample(
                n_experts=config["batch_size_task"])
            mc = []
            experts_sample_list = []
            for idx in range(config["batch_size_task"]):
                experts_sample = np.random.choice(experts_train, config["train_experts"]).tolist()
                experts_sample_list.append(experts_sample)
                exp_preds_cntx = []
                for idx_exp, expert in enumerate(experts_sample):
                    preds = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], None),
                                         device=device)
                    exp_preds_cntx.append(preds.unsqueeze(0))
                mc.append(torch.vstack(exp_preds_cntx))
            expert_cntx.mc = mc
            outputs = model(input, expert_cntx)  # [T,B,K+exp]

            loss = 0
            for idx, experts_sample in enumerate(experts_sample_list):
                collection_Ms = []
                for expert in experts_sample:
                    exp_preds = torch.tensor(expert(input, target, target_sparse), device=device)
                    costs = (exp_preds == target).int()
                    collection_Ms.append(costs)
                loss += loss_fn(outputs[idx], target, collection_Ms, n_classes + config["train_experts"])
            loss /= config["batch_size_task"]

            prec1 = accuracy(outputs.data[0, :, :n_classes], target, topk=(1,))[0]  # just measures clf accuracy
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            for optimizer in optimizer_lst:
                optimizer.zero_grad()
            loss.backward()
            for optimizer, scheduler in zip(optimizer_lst, scheduler_lst):
                optimizer.step()
                scheduler.step()
        elif config["l2d"] == 'MAML':
            expert_cntx = cntx_sampler.sample(n_experts=1)
            for optimizer in optimizer_lst:
                optimizer.zero_grad()

            loss_cum = 0
            for idx in range(config["batch_size_task"]):
                local_model = copy.deepcopy(model)
                local_model.train()

                experts_sample = np.random.choice(experts_train, config["train_experts"]).tolist()

                # freeze base network and classifier in train-time finetuning
                for param in local_model.params.base.parameters():
                    param.requires_grad = False
                for param in local_model.fc_clf.parameters():
                    param.requires_grad = False

                local_optim = torch.optim.SGD(local_model.parameters(), lr=lr_maml)
                local_optim.zero_grad()

                images_cntx = expert_cntx.xc[0]
                targets_cntx = expert_cntx.yc[0]
                cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse[0]

                collection_Ms = []
                for expert in experts_sample:
                    exp_preds_cntx = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], cntx_yc_sparse),
                                                  device=device)
                    costs = (exp_preds_cntx == targets_cntx).int()
                    collection_Ms.append(costs)

                for _ in range(n_steps_maml):
                    outputs = local_model(images_cntx)
                    loss = loss_fn(outputs, targets_cntx, collection_Ms, n_classes + config["train_experts"])
                    loss.backward()
                    local_optim.step()
                    local_optim.zero_grad()

                # unfreeze base network and classifier for global update
                for param in local_model.params.base.parameters():
                    param.requires_grad = True
                for param in local_model.fc_clf.parameters():
                    param.requires_grad = True

                collection_Ms = []
                for expert in experts_sample:
                    exp_preds = torch.tensor(expert(input, target, target_sparse), device=device)
                    costs = (exp_preds == target).int()
                    collection_Ms.append(costs)

                outputs = local_model(input)
                loss = loss_fn(outputs, target, collection_Ms, n_classes + config["train_experts"]) / config[
                    "batch_size_task"]

                loss.backward()

                for p_global, p_local in zip(model.parameters(), local_model.parameters()):
                    if p_global.grad is not None:
                        p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model

                loss_cum += loss

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data[:, :n_classes], target, topk=(1,))[0]  # just measures clf accuracy
            losses.update(loss_cum.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            for optimizer, scheduler in zip(optimizer_lst, scheduler_lst):
                optimizer.step()
                scheduler.step()

        else:
            experts_sample = np.random.choice(experts_train, config["train_experts"]).tolist()
            outputs = model(input)  # [B,K+1]
            collection_Ms = []
            for expert in experts_sample:
                exp_preds = torch.tensor(expert(input, target, target_sparse), device=device)
                costs = (exp_preds == target).int()
                collection_Ms.append(costs)
            loss = loss_fn(outputs, target, collection_Ms,n_classes+config["train_experts"])

            epoch_train_loss.append(loss.item())

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data[:, :n_classes], target, topk=(1,))[0]  # just measures clf accuracy
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            for optimizer in optimizer_lst:
                optimizer.zero_grad()
            loss.backward()
            for optimizer, scheduler in zip(optimizer_lst, scheduler_lst):
                optimizer.step()
                scheduler.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            loss=losses, top1=top1))




def train(model,
          train_loader,
          valid_loader,
          loss_fn,
          experts_train,
          cntx_sampler_train,
          cntx_sampler_eval,
          config,
          device):
    logger = get_logger(os.path.join(config["ckp_dir"], "train.log"))
    logger.info(f"p_out={config['p_out']}  seed={config['seed']}")
    logger.info(config)
    logger.info('No. of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    n_classes = config["n_classes"]

    model = model.to(device)
    cudnn.benchmark = True

    epochs = config["epochs"]
    lr_wrn = config["lr_wrn"]
    lr_clf_rej = config["lr_other"]

    # assuming epochs >= 50
    if epochs > 100:
        milestone_epoch = epochs - 50
    else:
        milestone_epoch = 50
    optimizer_base = torch.optim.SGD(model.params.base.parameters(),
                                     lr=lr_wrn,
                                     momentum=0.9,
                                     nesterov=True,
                                     weight_decay=config["weight_decay"])
    scheduler_base_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base,
                                                                       len(train_loader) * milestone_epoch,
                                                                       eta_min=lr_wrn / 1000)
    scheduler_base_constant = torch.optim.lr_scheduler.ConstantLR(optimizer_base, factor=1., total_iters=0)
    scheduler_base_constant.base_lrs = [lr_wrn / 1000 for _ in optimizer_base.param_groups]
    scheduler_base = torch.optim.lr_scheduler.SequentialLR(optimizer_base,
                                                           [scheduler_base_cosine, scheduler_base_constant],
                                                           milestones=[len(train_loader) * milestone_epoch])

    parameter_group = [{'params': model.params.clf.parameters()}]
    if config["l2d"] == "pop" or config["l2d"] == "EWA":
        parameter_group += [{'params': model.params.rej.parameters()}]
    optimizer_new = torch.optim.Adam(parameter_group, lr=lr_clf_rej)
    scheduler_new_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_new,
                                                                      len(train_loader) * milestone_epoch,
                                                                      eta_min=lr_clf_rej / 1000)
    scheduler_new_constant = torch.optim.lr_scheduler.ConstantLR(optimizer_new, factor=1., total_iters=0)
    scheduler_new_constant.base_lrs = [lr_clf_rej / 1000 for _ in optimizer_new.param_groups]
    scheduler_new = torch.optim.lr_scheduler.SequentialLR(optimizer_new, [scheduler_new_cosine, scheduler_new_constant],
                                                          milestones=[len(train_loader) * milestone_epoch])

    optimizer_lst = [optimizer_base, optimizer_new]
    scheduler_lst = [scheduler_base, scheduler_new]


    scoring_rule = config['scoring_rule']
    best_validation_loss = np.inf
    best_validation_acc = 0


    if config['l2d'] == 'single_maml':
        n_finetune_steps_eval = config['n_steps_maml']
    else:
        n_finetune_steps_eval = 0

    for epoch in range(1, epochs+1):
        train_epoch(train_loader,
                    model,
                    optimizer_lst,
                    scheduler_lst,
                    epoch,
                    experts_train,
                    loss_fn,
                    cntx_sampler_train,
                    n_classes,
                    config,
                    logger,
                    device,
                    config['n_steps_maml'],
                    config['lr_maml'])
        metrics = evaluate(model,
                           experts_train,
                           loss_fn,
                           cntx_sampler_eval,
                           n_classes,
                           valid_loader,
                           config,
                           device,
                           logger,
                           n_finetune_steps=n_finetune_steps_eval,
                           lr_finetune=config['lr_maml'])

        validation_loss = metrics['val_loss']
        validation_acc = metrics['sys_acc']

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(),
                       os.path.join(config["ckp_dir"], "best_loss.pt"))
        if validation_acc > best_validation_acc:
            best_validation_acc = validation_acc
            torch.save(model.state_dict(),
                       os.path.join(config["ckp_dir"], "best_acc.pt"))

        if (epoch + 1) == epochs:
            torch.save(model.state_dict(), os.path.join(
                config["ckp_dir"], "last_model.pt"))
    # Additionally save the whole config dict
    with open(os.path.join(config["ckp_dir"], config["experiment_name"] + ".json"), "w") as f:
        json.dump(config, f)



def eval(model, val_loader, test_loader, loss_fn, experts_test, experts_train, val_cntx_sampler, test_cntx_sampler, config, device):
    '''val_data and val_cntx_sampler are only used for single-expert finetuning'''
    model_name_list = ['last_model.pt', 'best_loss.pt', 'best_acc.pt']

    test_dir = f"{config['output_root']}/{config['dataset']}/{config['loss_type']}/" \
                        f"p{str(config['p_out'])}_seed{str(config['seed'])}/exp{config['train_experts']}" \
                        f"/l2d_{config['l2d']}/{config['time_stamp']}/{config['test_dataset']}"
    os.makedirs(test_dir, exist_ok=True)

    logger = get_logger(os.path.join(
        test_dir, "eval{}.log".format(1.0)))

    scoring_rule = config['scoring_rule']

    for name in model_name_list:
        print('Test', name)
        model_state_dict = torch.load(os.path.join(config["ckp_dir"], name),
                                      map_location=device)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        for budget in config["budget"]:
            if config["l2d"] != "MAML":
                test_cntx_sampler.reset()
                model.load_state_dict(copy.deepcopy(model_state_dict))
                evaluate(model, experts_test, loss_fn, test_cntx_sampler, config["n_classes"],
                         test_loader, config, device, logger, budget)
            if (config["l2d"] == 'MAML') or ((config["l2d"] == 'Multi') and config["finetune"]):
                logger = get_logger(os.path.join(test_dir, "eval{}_finetune.log".format(budget)))

                n_finetune_steps_lst = [n_steps for n_steps in config["n_finetune_steps"] if
                                        n_steps >= config["n_steps_maml"]] \
                    if (config["l2d"] == 'Multi') else config["n_finetune_steps"]
                lr_finetune_lst = [config["lr_maml"]] if (config["l2d"] == 'MAML') else config["lr_finetune"]

                steps_lr_comb = list(itertools.product(n_finetune_steps_lst, lr_finetune_lst))
                val_scores = []
                for (n_steps, lr) in steps_lr_comb:
                    print(f'no. finetune steps: {n_steps}  step size: {lr}')
                    val_cntx_sampler.reset()
                    model.load_state_dict(copy.deepcopy(model_state_dict))
                    metrics = evaluate(model, experts_train, loss_fn, val_cntx_sampler, config["n_classes"], val_loader,
                                       config, device, None, budget, n_steps, lr)
                    score = metrics[scoring_rule] if scoring_rule == 'val_loss' else -metrics[scoring_rule]
                    val_scores.append(score)
                idx = np.nanargmin(np.array(val_scores))
                best_finetune_steps, best_lr = steps_lr_comb[idx]
                test_cntx_sampler.reset()
                model.load_state_dict(copy.deepcopy(model_state_dict))
                metrics = evaluate(model, experts_test, loss_fn, test_cntx_sampler, config["n_classes"], test_loader,
                                   config, device, logger, budget, \
                                   best_finetune_steps, best_lr)


def main(config):

    dataset = config['dataset']
    test_dataset = config['test_dataset']
    batch_size_train = config['batch_size_train']
    batch_size_test = config["batch_size_test"]
    batch_size_valid = config["batch_size_valid"]
    download = config['download']
    as_rgb = config["as_rgb"]
    size = config["size"]
    resize = config["resize"]
    model_flag = config["model_flag"]

    set_seed(config["seed"])

    if config['mode'] == 'train':
        time_stamp = time.strftime('%y%m%d_%H%M%S')
        config['time_stamp'] = time_stamp
    else:
        time_stamp = config['time_stamp']
        if time_stamp is None:
            print("please assign the timestamp")
            return


    config["ckp_dir"] = f"{config['output_root']}/{dataset}/{config['loss_type']}/" \
                        f"p{str(config['p_out'])}_seed{str(config['seed'])}" \
                        f"/exp{config['train_experts']}/l2d_{config['l2d']}/{time_stamp}"
    os.makedirs(config["ckp_dir"], exist_ok=True)

    str_ids = config["gpu_ids"].split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
    # device = torch.device('cpu')

    info = INFO[dataset]
    config["task"] = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])
    config["n_classes"] = n_classes

    DataClass = getattr(medmnist, info['python_class'])
    TestDataClass= getattr(medmnist, INFO[test_dataset]['python_class'])

    print('==> Preparing data...')

    if config["resize"]:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb, size=size)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb, size=size)
    test_dataset = TestDataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb, size=size)


    kwargs = {'num_workers': 4, 'pin_memory': True}


    cntx_sampler_train = ContextSampler(train_dataset.imgs, train_dataset.labels, train_dataset.transform,
                                        None, \
                                        n_cntx_pts=config["n_cntx_pts"], device=device, **kwargs)

    prop_cntx = 0.2
    val_cntx_size = int(prop_cntx * len(val_dataset))
    val_data_cntx, val_data_trgt = torch.utils.data.random_split(val_dataset,
                                                                 [val_cntx_size, len(val_dataset) - val_cntx_size], \
                                                                 generator=torch.Generator().manual_seed(
                                                                     config["seed"]))

    test_cntx_size = int(prop_cntx * len(test_dataset))
    test_data_cntx, test_data_trgt = torch.utils.data.random_split(test_dataset,
                                                                   [test_cntx_size, len(test_dataset) - test_cntx_size], \
                                                                   generator=torch.Generator().manual_seed(
                                                                       config["seed"]))
    cntx_sampler_val = ContextSampler(images=val_data_cntx.dataset.imgs[val_data_cntx.indices],
                                      labels=val_data_cntx.dataset.labels[val_data_cntx.indices],
                                      transform=val_dataset.transform,
                                      labels_sparse=None,
                                      n_cntx_pts=config["n_cntx_pts"], device=device, **kwargs)
    cntx_sampler_test = ContextSampler(images=test_data_cntx.dataset.imgs[test_data_cntx.indices],
                                       labels=np.array(test_data_cntx.dataset.labels)[test_data_cntx.indices],
                                       transform=test_dataset.transform,
                                       labels_sparse=None,
                                       n_cntx_pts=config["n_cntx_pts"], device=device, **kwargs)


    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size_train,
                                   shuffle=True, **kwargs)
    val_loader = data.DataLoader(dataset=val_data_trgt,
                                 batch_size=batch_size_valid,
                                 shuffle=False, **kwargs)
    test_loader = data.DataLoader(dataset=test_data_trgt,
                                  batch_size=batch_size_test,
                                  shuffle=False, **kwargs)

    print('==> Building and training model...')

    if model_flag == 'resnet18':
        resnet_base = resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes, norm_type=config["norm_type"])
    elif model_flag == 'resnet50':
        resnet_base = resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels, num_classes=n_classes, norm_type=config["norm_type"])
    else:
        raise NotImplementedError

    n_features = resnet_base.n_features

    with_softmax = False
    if config["loss_type"] == 'softmax':
        loss_fn = softmax
        with_softmax = True
    else:  # ova
        loss_fn = ova

    with_attn = False
    config_tokens = config["l2d"].split("_")
    if len(config_tokens) > 1:
        if config_tokens[1] == 'attn':
            with_attn = True
        config["l2d"] = config_tokens[0]

    if config["warmstart"]:
        if config["dataset"] == "cifar10" or config["dataset"] == "cifar20_100":
            fn_aug = '' if config['norm_type']=='batchnorm' else '_frn'
            warmstart_path = f"./pretrained/{config['dataset']}{fn_aug}/seed{str(config['seed'])}/default.pt"
        else:
            warmstart_path = f"./pretrained/{config['dataset']}/seed{str(config['seed'])}/default.pt"
        if not os.path.isfile(warmstart_path):
            raise FileNotFoundError('warmstart model checkpoint not found')
        resnet_base.load_state_dict(torch.load(warmstart_path, map_location=device))
        resnet_base = resnet_base.to(device)

    if config["l2d"] == "EWA":
        model = ExpertWiseAggregator(resnet_base, num_classes=int(config["n_classes"]),
                                                      n_features=n_features, \
                                                      with_attn=with_attn, with_softmax=with_softmax,
                                                      decouple=config["decouple"], \
                                                      depth_embed=config["depth_embed"],
                                                      depth_rej=config["depth_reject"])
    elif config["l2d"] == "pop":
        model = ClassifierRejectorWithContextEmbedder(resnet_base, num_classes=int(config["n_classes"]),
                                                      n_features=n_features, \
                                                      n_experts=config["train_experts"],
                                                      with_attn=with_attn, with_softmax=with_softmax,
                                                      decouple=config["decouple"], \
                                                      depth_embed=config["depth_embed"],
                                                      depth_rej=config["depth_reject"])
    else:
        model = ClassifierRejector(resnet_base, num_classes=int(config["n_classes"]), n_features=n_features,
                                   n_experts=config["train_experts"], with_softmax=with_softmax, \
                                   decouple=config["decouple"])

    config["n_experts"] = 10  # assume exactly divisible by 2
    experts_test, experts_train = [], []
    can_class_train, can_class_test = [], []
    pop_dict = {"pop1": [i for i in range(0,5)],
                "pop2": [i for i in range(5,11)]}

    for pop_k, pop_v in pop_dict.items():
        train_exp = []
        for _ in range(config["n_experts"]):  # train
            class_num = np.random.randint(1, len(pop_v)+1)
            class_oracle = np.random.choice(pop_v, class_num, replace=False)
            expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0,
                                            p_out=config['p_out'])
            train_exp.append(expert)
            can_class_train.append(class_oracle.tolist())
        experts_train += train_exp
        experts_test += train_exp[:config["n_experts"] // 2]  # pick 50% experts from experts_train (order not matter)
    for pop_k, pop_v in pop_dict.items():
        for _ in range(config["n_experts"] // 2):  # then sample 50% new experts
            class_num = np.random.randint(1, len(pop_v) + 1)
            class_oracle = np.random.choice(pop_v, class_num, replace=False)
            expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0,
                                            p_out=config['p_out'])
            experts_test.append(expert)
            can_class_test.append(class_oracle.tolist())
    print("Test Experts => ", can_class_train)
    print("Train Experts => ", can_class_test)


    if config["mode"] == 'train':
        train(model, train_loader, val_loader, loss_fn, experts_train, cntx_sampler_train,
              cntx_sampler_val, config, device)
        eval(model, val_loader, test_loader, loss_fn, experts_test, experts_train,cntx_sampler_val,
             cntx_sampler_test, config, device)
    else:  # evaluation on test data
        eval(model, val_loader, test_loader, loss_fn, experts_test, experts_train,cntx_sampler_val,
             cntx_sampler_test, config, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Base set
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--dataset', type=str, default='organamnist',
                        choices=["chestmnist", "pneumoniamnist", "organamnist", "organcmnist", "organsmnist"])
    parser.add_argument('--test_dataset', type=str, default='organamnist',
                        choices=["chestmnist", "pneumoniamnist", "organamnist", "organcmnist", "organsmnist"])
    parser.add_argument("--epochs", type=int, default=100)  # @@ cifar,gtsrb:150 ham:100
    parser.add_argument('--gpu_ids', default='1', type=str)

    parser.add_argument('--norm_type', choices=['batchnorm', 'frn'], default='frn')
    parser.add_argument("--p_out", type=float, default=0.1)  # [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    parser.add_argument('--loss_type', choices=['softmax', 'ova'], default='softmax')

    ## MedMNIST2D
    parser.add_argument('--output_root', type=str, default='./output',
                        help='output root, where to save models and results')
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

    ## Training options06
    parser.add_argument('--l2d', choices=['EWA', 'EWA_attn', 'pop', 'pop_attn', 'Multi', 'MAML'], default='MAML')
    parser.add_argument("--train_experts", type=int, default=3)
    parser.add_argument("--batch_size_train", type=int, default=128)  # @@ cifar,gtsrb:128 ham:64
    parser.add_argument("--batch_size_valid", type=int, default=8)
    parser.add_argument("--lr_wrn", type=float, default=1e-2, help="learning rate for wrn.")
    parser.add_argument("--lr_other", type=float, default=1e-3, help="learning rate for non-wrn model components.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)  # @@ cifar,ham:5e-4 gtsrb:1e-3
    parser.add_argument('--warmstart', action='store_true')
    parser.set_defaults(warmstart=True)
    parser.add_argument('--decouple', action='store_true')
    parser.set_defaults(decouple=False)
    parser.add_argument('--scoring_rule', choices=['val_loss', 'sys_acc'], default='val_loss')
    parser.add_argument("--experiment_name", type=str, default="default",
                        help="specify the experiment name. Checkpoints will be saved with this name.")

    ## Meta Methods
    parser.add_argument("--n_cntx_pts", type=int, default=50)  # @@ cifar20:100 other:50
    # MAML
    parser.add_argument("--batch_size_task", type=int, default=4)
    parser.add_argument('--n_steps_maml', type=int, default=2)  # @@ cifar: (2,0.1) gtsrb(5,0.1) ham(2,0.01)
    parser.add_argument('--lr_maml', type=float, default=0.1)  # @@ cifar: (2,0.1) gtsrb(5,0.1) ham(2,0.01)
    # Attention
    parser.add_argument("--depth_embed", type=int, default=6)  # @@ gtsrb:(5,3) other:(6,4)
    parser.add_argument("--depth_reject", type=int, default=4)

    ## EVAL
    parser.add_argument("--batch_size_test", type=int, default=1)
    parser.add_argument('--budget', nargs='+', type=float, default=[1.0])  # default=[0.01,0.02,0.05,0.1,0.2,0.5]
    parser.add_argument('--finetune', action='store_true')
    parser.set_defaults(finetune=True)
    parser.add_argument('--n_finetune_steps', nargs='+', type=int, default=[1, 2, 5, 10])
    parser.add_argument('--lr_finetune', nargs='+', type=float, default=[1e-1, 1e-2])
    parser.add_argument('--time_stamp', type=str, default='250118_140041')

    config = parser.parse_args().__dict__
    main(config)


