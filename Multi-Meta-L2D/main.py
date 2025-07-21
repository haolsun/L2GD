from attrdict import AttrDict
import math
import random
import argparse
import os
import shutil
import time
import json
import functools
import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity
import time

# local imports
from lib.utils import AverageMeter, accuracy, get_logger
from lib.losses import cross_entropy, ova, softmax
from lib.experts import SyntheticExpertOverlap, Cifar20SyntheticExpert
from lib.modules_multiexp import ClassifierRejector, ClassifierRejectorWithContextEmbedder_EWA, ClassifierRejectorWithContextEmbedder_Pop
from lib.resnet224 import ResNet34
from lib.resnet import resnet20
from lib.datasets import load_cifar, load_ham10000, load_gtsrb, ContextSampler
from lib.wideresnet import WideResNetBase


device = torch.device("cuda:0")


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
    #  === Individual Expert Accuracies === #

    model.eval()  # Crucial for networks with batchnorm layers!
    if config["l2d"] == 'maml':
        model.train()
    is_finetune = ((config["l2d"] == 'multi_L2D') or (
        config["l2d"] == 'maml')) and (n_finetune_steps > 0)
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
        if len(data) == 2:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            labels_sparse = None
        else:
            images, labels, labels_sparse = data
            images, labels, labels_sparse = images.to(
                device), labels.to(device), labels_sparse.to(device)

        experts_sample = np.random.choice(
            experts_test, config["train_experts"]).tolist()

        # sample expert predictions for context
        if config["l2d"] == 'EWA':
            expert_cntx = cntx_sampler.sample(n_experts=config["train_experts"])
            exp_preds_cntx = []
            for idx_exp, expert in enumerate(experts_sample):
                cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse[idx_exp]
                preds = torch.tensor(expert(expert_cntx.xc[idx_exp], expert_cntx.yc[idx_exp], cntx_yc_sparse),
                                    device=device)
                exp_preds_cntx.append(preds.unsqueeze(0))
            expert_cntx.mc = torch.vstack(exp_preds_cntx)
        else:
            expert_cntx = cntx_sampler.sample(n_experts=1)
            cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse.squeeze(0)
            collection_Ms = []
            exp_preds_cntx_list = []
            mc = []
            for expert in experts_sample:
                exp_preds_cntx = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], cntx_yc_sparse),
                                            device=device)
                exp_preds_cntx_list.append(exp_preds_cntx.unsqueeze(0))
                costs = (exp_preds_cntx == expert_cntx.yc.squeeze(0)).int()
                collection_Ms.append(costs)
            mc.append(torch.vstack(exp_preds_cntx_list))
            expert_cntx.mc = mc

            if is_finetune:
                model.train()
                images_cntx = expert_cntx.xc.squeeze(0)
                targets_cntx = expert_cntx.yc.squeeze(0)

                # NB: could freeze base network like finetuning in train_epoch()
                for _ in range(n_finetune_steps):
                    outputs_cntx = model(images_cntx)
                    loss = loss_fn(outputs_cntx, targets_cntx,
                                collection_Ms, n_classes+config["train_experts"])
                    model.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        for param in model.params.clf.parameters():
                            new_param = param - lr_finetune * param.grad
                            param.copy_(new_param)
                # finetuning on multi-expert, use running batch statistics for eval
                if config["l2d"] == 'multi_L2D':
                    model.eval()

        with torch.no_grad():
            # removes expert context based on coin flip
            coin_flip = np.random.binomial(1, p_cntx_inclusion)
            if coin_flip == 0:
                expert_cntx = None

            if config["l2d"] == 'pop':
                outputs = model(images, expert_cntx).squeeze(0)
            elif config["l2d"] == 'EWA': 
                outputs = model(images, expert_cntx)
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
                if config["l2d"] == 'multi_L2D':
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
    #  === Individual Expert Accuracies === #
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


def train_epoch(iters,
                train_loader,
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
        if len(data) == 2:
            input, target = data
            input, target = input.to(device), target.to(device)
            target_sparse = None
        else:
            input, target, target_sparse = data  # ignore additional labels
            input, target, target_sparse = input.to(
                device), target.to(device), target_sparse.to(device)
            
        # For MAML: need to do backprop once at start to initialize grads
        if (i == 0) and (config["l2d"] == 'maml'):

            collection_Ms = []
            for expert in range(config["train_experts"]):
                costs = torch.zeros_like(target, device=target.device)
                collection_Ms.append(costs)

            outputs = model(input)
            loss = loss_fn(outputs, target, collection_Ms,
                           n_classes+config["train_experts"])  # loss per expert
            loss.backward()
            model.zero_grad()

        if config["l2d"] == 'maml':
            expert_cntx = cntx_sampler.sample(n_experts=1)
            for optimizer in optimizer_lst:
                optimizer.zero_grad()

            loss_cum = 0
            for idx in range(config["meta_batch_size"]):
                local_model = copy.deepcopy(model)
                local_model.train()

                experts_sample = np.random.choice(
                    experts_train, config["train_experts"]).tolist()

                # freeze base network and classifier in train-time finetuning
                for param in local_model.params.base.parameters():
                    param.requires_grad = False
                for param in local_model.fc_clf.parameters():
                    param.requires_grad = False

                local_optim = torch.optim.SGD(
                    local_model.parameters(), lr=lr_maml)
                local_optim.zero_grad()

                images_cntx = expert_cntx.xc[0]
                targets_cntx = expert_cntx.yc[0]
                cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse[
                    0]

                collection_Ms = []
                for expert in experts_sample:
                    exp_preds_cntx = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], cntx_yc_sparse),
                                                  device=device)
                    costs = (exp_preds_cntx == targets_cntx).int()
                    collection_Ms.append(costs)

                for _ in range(n_steps_maml):
                    outputs = local_model(images_cntx)
                    loss = loss_fn(outputs, targets_cntx, collection_Ms,
                                   n_classes+config["train_experts"])
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
                    exp_preds = torch.tensor(
                        expert(input, target, target_sparse), device=device)
                    costs = (exp_preds == target).int()
                    collection_Ms.append(costs)

                outputs = local_model(input)
                loss = loss_fn(outputs, target, collection_Ms, n_classes +
                               config["train_experts"]) / config["meta_batch_size"]

                loss.backward()

                for p_global, p_local in zip(model.parameters(), local_model.parameters()):
                    if p_global.grad is not None:
                        # First-order approx. -> add gradients of finetuned and base model
                        p_global.grad += p_local.grad

                loss_cum += loss

            epoch_train_loss.append(loss_cum.item())

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data[:, :n_classes], target, topk=(1,))[
                0]  # just measures clf accuracy
            losses.update(loss_cum.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            for optimizer, scheduler in zip(optimizer_lst, scheduler_lst):
                optimizer.step()
                scheduler.step()
        elif config["l2d"] == 'EWA':
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

            outputs = model(input, expert_cntx)  # [B,K+n_exp]

            collection_Ms = []
            for expert in experts_sample:
                exp_preds = torch.tensor(
                    expert(input, target, target_sparse), device=device)
                costs = (exp_preds == target).int()
                collection_Ms.append(costs)
            loss = loss_fn(outputs, target, collection_Ms,
                            n_classes + config["train_experts"])

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data[:, :n_classes], target, topk=(1,))[
                0]  # just measures clf accuracy
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            for optimizer in optimizer_lst:
                optimizer.zero_grad()
            loss.backward()
            for optimizer, scheduler in zip(optimizer_lst, scheduler_lst):
                optimizer.step()
                scheduler.step()
        elif config["l2d"] == 'pop':
            expert_cntx = cntx_sampler.sample(
                n_experts=config["meta_batch_size"])
            mc = []

            experts_sample_list = []

            for idx in range(config["meta_batch_size"]):
                experts_sample = np.random.choice(
                    experts_train, config["train_experts"]).tolist()
                experts_sample_list.append(experts_sample)
                exp_preds_cntx = []
                for idx_exp, expert in enumerate(experts_sample):
                    cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse[
                        0]
                    preds = torch.tensor(expert(expert_cntx.xc[0], expert_cntx.yc[0], cntx_yc_sparse),
                                         device=device)
                    exp_preds_cntx.append(preds.unsqueeze(0))
                mc.append(torch.vstack(exp_preds_cntx))
            expert_cntx.mc = mc
            outputs = model(input, expert_cntx)

            loss = 0
            for idx, experts_sample in enumerate(experts_sample_list):
                collection_Ms = []
                for expert in experts_sample:
                    exp_preds = torch.tensor(
                        expert(input, target, target_sparse), device=device)
                    costs = (exp_preds == target).int()
                    collection_Ms.append(costs)
                loss += loss_fn(outputs[idx], target, collection_Ms,
                                n_classes + config["train_experts"])
            loss /= config["meta_batch_size"]

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data[0, :, :n_classes], target, topk=(1,))[
                0]  # just measures clf accuracy
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            for optimizer in optimizer_lst:
                optimizer.zero_grad()
            loss.backward()
            for optimizer, scheduler in zip(optimizer_lst, scheduler_lst):
                optimizer.step()
                scheduler.step()
        else:
            experts_sample = np.random.choice(
                experts_train, config["train_experts"]).tolist()
            outputs = model(input)  # [B,K+1]
            collection_Ms = []
            for expert in experts_sample:
                exp_preds = torch.tensor(
                    expert(input, target, target_sparse), device=device)
                costs = (exp_preds == target).int()
                collection_Ms.append(costs)
            loss = loss_fn(outputs, target, collection_Ms,
                           n_classes+config["train_experts"])

            epoch_train_loss.append(loss.item())

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data[:, :n_classes], target, topk=(1,))[
                0]  # just measures clf accuracy
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
        iters += 1

        # if i % 10 == 0:
        if i % 10 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            loss=losses, top1=top1))

    return iters, np.average(epoch_train_loss)


def train(model,
          train_dataset,
          validation_dataset,
          loss_fn,
          experts_train,
          experts_test,
          cntx_sampler_train,
          cntx_sampler_eval,
          config):
    logger = get_logger(os.path.join(config["ckp_dir"], "train.log"))
    logger.info(f"p_out={config['p_out']}  seed={config['seed']}")
    logger.info(config)
    logger.info('No. of parameters: {}'.format(sum(p.numel()
                for p in model.parameters() if p.requires_grad)))
    n_classes = config["n_classes"]
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["train_batch_size"], shuffle=True,
                                               **kwargs)  # drop_last=True
    valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=config["val_batch_size"], shuffle=False,
                                               **kwargs)  # shuffle=True, drop_last=True

    # model = nn.DataParallel(model, device_ids=[0,1,2])
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
                                                                       len(train_loader) *
                                                                       milestone_epoch,
                                                                       eta_min=lr_wrn / 1000)
    scheduler_base_constant = torch.optim.lr_scheduler.ConstantLR(
        optimizer_base, factor=1., total_iters=0)
    scheduler_base_constant.base_lrs = [
        lr_wrn / 1000 for _ in optimizer_base.param_groups]
    scheduler_base = torch.optim.lr_scheduler.SequentialLR(optimizer_base,
                                                           [scheduler_base_cosine,
                                                               scheduler_base_constant],
                                                           milestones=[len(train_loader) * milestone_epoch])

    parameter_group = [{'params': model.params.clf.parameters()}]
    if (config["l2d"] == "pop") or (config["l2d"] == "EWA"):
        parameter_group += [{'params': model.params.rej.parameters()}]
    optimizer_new = torch.optim.Adam(parameter_group, lr=lr_clf_rej)
    scheduler_new_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_new,
                                                                      len(train_loader) *
                                                                      milestone_epoch,
                                                                      eta_min=lr_clf_rej / 1000)
    scheduler_new_constant = torch.optim.lr_scheduler.ConstantLR(
        optimizer_new, factor=1., total_iters=0)
    scheduler_new_constant.base_lrs = [
        lr_clf_rej / 1000 for _ in optimizer_new.param_groups]
    scheduler_new = torch.optim.lr_scheduler.SequentialLR(optimizer_new, [scheduler_new_cosine, scheduler_new_constant],
                                                          milestones=[len(train_loader) * milestone_epoch])

    optimizer_lst = [optimizer_base, optimizer_new]
    scheduler_lst = [scheduler_base, scheduler_new]

    scoring_rule = config['scoring_rule']
    best_validation_loss = np.inf
    best_validation_acc = 0
    iters = 0

    if config['l2d'] == 'maml':
        n_finetune_steps_eval = config['n_steps_maml']
    else:
        n_finetune_steps_eval = 0

    for epoch in range(1, epochs+1):
        iters, train_loss = train_epoch(iters,
                                        train_loader,
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
                                        config['n_steps_maml'],
                                        config['lr_maml'])
        metrics = evaluate(model,
                           experts_train,
                           loss_fn,
                           cntx_sampler_eval,
                           n_classes,
                           valid_loader,
                           config,
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



def eval(model, val_data, test_data, loss_fn, experts_test, experts_train, val_cntx_sampler, test_cntx_sampler, config):
    '''val_data and val_cntx_sampler are only used for single-expert finetuning'''
    # model_name_list = ['last_model.pt', 'best_loss.pt', 'best_acc.pt']
    model_name_list = ['best_acc.pt']

    kwargs = {'num_workers': 0, 'pin_memory': True}
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=config["val_batch_size"], shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config["test_batch_size"], shuffle=False, **kwargs)
    
    scoring_rule = 'sys_acc'
    for name in model_name_list:
        print('Test', name)
        model_state_dict = torch.load(os.path.join(config["ckp_dir"], name),
                                        map_location=device)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        for budget in config["budget"]:
            if config["l2d"] not in ['maml']:
                test_cntx_sampler.reset()
                logger = get_logger(os.path.join(
                    config["ckp_dir"], "eval{}.log".format(budget)))
                model.load_state_dict(copy.deepcopy(model_state_dict))
                evaluate(model, experts_test, loss_fn, test_cntx_sampler, config["n_classes"], test_loader, config, logger,
                            budget)
            if (config["l2d"] == 'maml') or ((config["l2d"] == 'multi_L2D') and config["finetune_single"]):
                logger = get_logger(os.path.join(
                    config["ckp_dir"], "eval{}_finetune.log".format(budget)))
                n_finetune_steps_lst = [n_steps for n_steps in config["n_finetune_steps"] if
                                          n_steps >= config["n_steps_maml"]] \
                      if (config["l2d"] == 'maml') else config["n_finetune_steps"]
                lr_finetune_lst = [config["lr_maml"]] if (
                       config["l2d"] == 'maml') else config["lr_finetune"]

                steps_lr_comb = list(itertools.product(n_finetune_steps_lst, lr_finetune_lst))
                val_scores = []
                for (n_steps, lr) in steps_lr_comb:
                    print(f'no. finetune steps: {n_steps}  step size: {lr}')
                    val_cntx_sampler.reset()
                    model.load_state_dict(copy.deepcopy(model_state_dict))
                    metrics = evaluate(model, experts_train, loss_fn, val_cntx_sampler, config["n_classes"], val_loader,
                                               config, None, budget,
                                               n_steps, lr)
                    score = metrics[scoring_rule] if scoring_rule == 'val_loss' else - \
                                metrics[scoring_rule]
                    val_scores.append(score)
                    idx = np.nanargmin(np.array(val_scores))
                    best_finetune_steps, best_lr = steps_lr_comb[idx]
                
                test_cntx_sampler.reset()
                model.load_state_dict(copy.deepcopy(model_state_dict))
                metrics = evaluate(model, experts_test, loss_fn, test_cntx_sampler, config["n_classes"], test_loader,
                                    config, logger, budget,
                                    best_finetune_steps, best_lr)



def main(config):

    set_seed(config["seed"])
    config[
        "ckp_dir"] = f"./run_results/{config['dataset']}/trainexp{config['train_experts']}/l2d_{config['l2d']}/{config['loss_type']}/p{str(config['p_out'])}_seed{str(config['seed'])}"
    os.makedirs(config["ckp_dir"], exist_ok=True)

    if config["dataset"] == 'cifar100_SC':
        config["n_classes"] = 20
        train_data, val_data, test_data = load_cifar(
            variety='20_100', data_aug=True, seed=config["seed"])
        resnet_base = WideResNetBase(
            depth=28, n_channels=3, widen_factor=4, dropRate=0.0, norm_type=config["norm_type"])
        n_features = resnet_base.nChannels
    elif config["dataset"] == 'cifar10':
        config["n_classes"] = 10
        train_data, val_data, test_data = load_cifar(
            variety='10', data_aug=False, seed=config["seed"])
        resnet_base = WideResNetBase(
            depth=28, n_channels=3, widen_factor=2, dropRate=0.0, norm_type=config["norm_type"])
        n_features = resnet_base.nChannels
    elif config["dataset"] == 'ham10000':
        config["n_classes"] = 7
        train_data, val_data, test_data = load_ham10000()
        resnet_base = ResNet34()
        n_features = resnet_base.n_features
    elif config["dataset"] == 'gtsrb':
        config["n_classes"] = 43
        train_data, val_data, test_data = load_gtsrb()
        resnet_base = resnet20(norm_type=config["norm_type"])
        n_features = resnet_base.n_features
    else:
        raise ValueError('dataset unrecognised')

    with_softmax = False
    if config["loss_type"] == 'softmax':
        loss_fn = softmax
        with_softmax = True
    else:  # ova
        loss_fn = ova

    with_attn = False
    config_tokens = config["l2d"].split("_")
    if (len(config_tokens) > 1) and ((config_tokens[0] == 'pop') or (config_tokens[0] == 'EWA')):
        if config_tokens[1] == 'attn':
            with_attn = True
        config["l2d"] = config_tokens[0]

    if config["warmstart"]:
        if config["dataset"] == "cifar10" or config["dataset"] == "cifar100_SC":
            fn_aug = '' if config['norm_type'] == 'batchnorm' else '_frn'
            warmstart_path = f"./pretrained/{config['dataset']}{fn_aug}/seed{str(config['seed'])}/default.pt"
        else:
            warmstart_path = f"./pretrained/{config['dataset']}/seed{str(config['seed'])}/default.pt"
        if not os.path.isfile(warmstart_path):
            raise FileNotFoundError('warmstart model checkpoint not found')
        resnet_base.load_state_dict(torch.load(
            warmstart_path, map_location=device))
        resnet_base = resnet_base.to(device)

    if config["l2d"] == "pop":
        model = ClassifierRejectorWithContextEmbedder_Pop(resnet_base, num_classes=int(config["n_classes"]),
                                                      n_features=n_features,
                                                      n_experts=config["train_experts"],
                                                      dim_hid=config['dim_hid'],
                                                      depth_embed=config["depth_embed"],
                                                      depth_rej=config["depth_reject"],
                                                      with_attn=with_attn, with_softmax=with_softmax,decouple=config["decouple"])
    elif config["l2d"] == "EWA":
        model = ClassifierRejectorWithContextEmbedder_EWA(resnet_base, num_classes=int(config["n_classes"]),
                                                          n_features=n_features,
                                                          n_experts=config["train_experts"],
                                                          dim_hid=config['dim_hid'],
                                                          depth_embed=config["depth_embed"],
                                                          depth_rej=config["depth_reject"],
                                                          with_attn=with_attn, with_softmax=with_softmax, decouple=config["decouple"])
    else:
        model = ClassifierRejector(resnet_base, num_classes=int(config["n_classes"]), n_features=n_features,
                                   n_experts=config["train_experts"], with_softmax=with_softmax,
                                   decouple=config["decouple"])

    # ==================================================================
    # | -- Set Experts
    if config["dataset"] == 'cifar10':
        config["n_experts"] = 6  # assume exactly divisible by 2
        pop_dict = {"pop1": [0, 1, 2],
                    "pop2": [3, 4, 5],
                    "pop3": [6, 7, 8, 9]}
        experts_test = []
        experts_train = []
        can_class_train = []
        can_class_test = []
        p_in = 1.0
        for pop_k, pop_v in pop_dict.items():
            train_exp = []
            for _ in range(config["n_experts"]):  # train
                can_class_num = np.random.randint(1, len(pop_v)+1)
                can_class_list = np.random.choice(pop_v, can_class_num, replace=False)
                can_class_train.append(can_class_list)
                expert = SyntheticExpertOverlap(classes_oracle=can_class_list, n_classes=config["n_classes"], p_in=p_in,
                                            p_out=config["p_out"])
                train_exp.append(expert)

            experts_train += train_exp
            experts_test += train_exp[
                            :config["n_experts"] // 2]  # pick 50% experts from experts_train (order not matter)

        for pop_k, pop_v in pop_dict.items():
            for _ in range(config["n_experts"] // 2):  # then sample 50% new experts
                can_class_num = np.random.randint(1, len(pop_v)+1)
                can_class_list = np.random.choice(pop_v, can_class_num, replace=False)
                can_class_test.append(can_class_list)
                expert = SyntheticExpertOverlap(classes_oracle=can_class_list, n_classes=config["n_classes"], p_in=p_in,
                                            p_out=config["p_out"])
                experts_test.append(expert)

    elif config["dataset"] == 'cifar20_100':
        config["n_experts"] = 10  # assume exactly divisible by 2
        pop_dict = {"pop1": [i for i in range(0, 10)],
                    "pop2": [i for i in range(10, 20)]}
        experts_test = []
        experts_train = []
        can_class_train = []
        can_class_test = []
        n_oracle_superclass = 4
        # 3 or 4 here. Affects gap between {single,pop,pop_attn}
        n_oracle_subclass = 3
        p_in = 1.0
        for pop_k, pop_v in pop_dict.items():
            train_exp = []
            for _ in range(config["n_experts"]):  # train
                classes_coarse = np.random.choice(
                    pop_v, n_oracle_superclass, replace=False)
                expert = Cifar20SyntheticExpert(classes_coarse, n_classes=config["n_classes"], p_in=1.0,
                                                p_out=config['p_out'],
                                                n_oracle_subclass=n_oracle_subclass)
                can_class_train.append(classes_coarse)
                train_exp.append(expert)
            experts_train += train_exp
            experts_test += train_exp[
                :config["n_experts"] // 2]  # pick 50% experts from experts_train (order not matter)

        for pop_k, pop_v in pop_dict.items():
            for _ in range(config["n_experts"] // 2):  # then sample 50% new experts
                classes_coarse = np.random.choice(
                    pop_v, size=n_oracle_superclass, replace=False)
                expert = Cifar20SyntheticExpert(classes_coarse, n_classes=config["n_classes"], p_in=1.0,
                                                p_out=config['p_out'],
                                                n_oracle_subclass=n_oracle_subclass)
                can_class_test.append(classes_coarse)
                experts_test.append(expert)
    elif config["dataset"] == 'gtsrb':
        config["n_experts"] = 10  # assume exactly divisible by 2
        pop_dict = {"pop1": [i for i in range(0, 8)],
                    "pop2": [i for i in range(8, 16)],
                    "pop3": [i for i in range(16, 25)],
                    "pop4": [i for i in range(25, 34)],
                    "pop5": [i for i in range(34, 43)]}
        experts_train = []
        experts_test = []
        can_class_train = []
        can_class_test = []
        p_in = 1.0
        for pop_k, pop_v in pop_dict.items():
            train_exp = []
            for _ in range(config["n_experts"]):  # train
                can_class_num = np.random.randint(1, len(pop_v)+1)
                can_class_list = np.random.choice(
                    pop_v, can_class_num, replace=False)
                expert = SyntheticExpertOverlap(classes_oracle=can_class_list, n_classes=config["n_classes"], p_in=p_in,
                                                p_out=config["p_out"])
                train_exp.append(expert)
                can_class_train.append(can_class_list.tolist())
            experts_train += train_exp
            experts_test += train_exp[
                :config["n_experts"] // 2]  # pick 50% experts from experts_train (order not matter)

        for pop_k, pop_v in pop_dict.items():
            for _ in range(config["n_experts"] // 2):  # then sample 50% new experts
                can_class_num = np.random.randint(1, len(pop_v)+1)
                can_class_list = np.random.choice(
                    pop_v, can_class_num, replace=False)
                expert = SyntheticExpertOverlap(classes_oracle=can_class_list, n_classes=config["n_classes"], p_in=p_in,
                                                p_out=config["p_out"])
                experts_test.append(expert)
                can_class_test.append(can_class_list.tolist())
    elif config["dataset"] == 'ham10000':
        config["n_experts"] = 10  # assume exactly divisible by 2
        pop_dict = {"pop1": [0, 1, 2],
                    "pop2": [3, 4, 5, 6]}
        experts_test = []
        experts_train = []
        can_class_train = []
        can_class_test = []
        p_in = 1.0
        for pop_k, pop_v in pop_dict.items():
            train_exp = []
            for _ in range(config["n_experts"]):  # train
                can_class_num = np.random.randint(1, len(pop_v)+1)
                can_class_list = np.random.choice(
                    pop_v, can_class_num, replace=False)
                expert = SyntheticExpertPop(class_list=can_class_list, n_classes=config["n_classes"], p_in=p_in,
                                            p_out=config["p_out"])
                train_exp.append(expert)
                can_class_train.append(can_class_list.tolist())
            experts_train += train_exp
            experts_test += train_exp[
                :config["n_experts"] // 2]  # pick 50% experts from experts_train (order not matter)

        for pop_k, pop_v in pop_dict.items():
            for _ in range(config["n_experts"] // 2):  # then sample 50% new experts
                can_class_num = np.random.randint(1, len(pop_v)+1)
                can_class_list = np.random.choice(
                    pop_v, can_class_num, replace=False)
                expert = SyntheticExpertPop(class_list=can_class_list, n_classes=config["n_classes"], p_in=p_in,
                                            p_out=config["p_out"])
                experts_test.append(expert)
                can_class_test.append(can_class_list.tolist())

    # Context sampler train-time: just take from full train set (potentially with data augmentation)
    kwargs = {'num_workers': 4, 'pin_memory': True}

    cntx_sampler_train = ContextSampler(train_data.data, train_data.targets, train_data.transform,
                                        train_data.targets_sparse,
                                        n_cntx_pts=config["n_cntx_pts"], device=device, **kwargs)

    # Context sampler val/test-time: partition val/test sets
    prop_cntx = 0.2
    val_cntx_size = int(prop_cntx * len(val_data))  
    val_data_cntx, val_data_trgt = torch.utils.data.random_split(val_data,
                                                                 [val_cntx_size, len(
                                                                     val_data) - val_cntx_size],
                                                                 generator=torch.Generator().manual_seed(
                                                                     config["seed"]))
    test_cntx_size = int(prop_cntx * len(test_data))  
    test_data_cntx, test_data_trgt = torch.utils.data.random_split(test_data,
                                                                   [test_cntx_size, len(
                                                                       test_data) - test_cntx_size],
                                                                   generator=torch.Generator().manual_seed(
                                                                       config["seed"]))
    cntx_sampler_val = ContextSampler(images=val_data_cntx.dataset.data[val_data_cntx.indices],
                                      labels=val_data_cntx.dataset.targets[val_data_cntx.indices],
                                      transform=val_data.transform,
                                      labels_sparse=val_data_cntx.dataset.targets_sparse[val_data_cntx.indices] if
                                      config["dataset"] == 'cifar100_SC' else None,
                                      n_cntx_pts=config["n_cntx_pts"], device=device, **kwargs)
    cntx_sampler_test = ContextSampler(images=test_data_cntx.dataset.data[test_data_cntx.indices],
                                       labels=np.array(test_data_cntx.dataset.targets)[test_data_cntx.indices],
                                       transform=test_data.transform,
                                       labels_sparse=test_data_cntx.dataset.targets_sparse[test_data_cntx.indices] if
                                       config["dataset"] == 'cifar100_SC' else None,
                                       n_cntx_pts=config["n_cntx_pts"], device=device, **kwargs)

    if config["mode"] == 'train':
        train(model, train_data, val_data_trgt, loss_fn, experts_train, experts_test, cntx_sampler_train,
              cntx_sampler_val, config)
        eval(model, val_data_trgt, test_data_trgt, loss_fn,
             experts_test, experts_train, cntx_sampler_val, cntx_sampler_test, config)
    else:  # evaluation on test data
        eval(model, val_data_trgt, test_data_trgt, loss_fn,
             experts_test, experts_train, cntx_sampler_val, cntx_sampler_test, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Base set
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset", choices=["cifar10", "cifar100_SC", "ham10000", "gtsrb"], default="cifar10")
    parser.add_argument(
        '--norm_type', choices=['batchnorm', 'frn'], default='frn')
    # [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    parser.add_argument("--p_out", type=float, default=0.1)
    parser.add_argument(
        '--l2d', choices=['EWA', 'EWA_attn', 'pop', 'pop_attn', 'multi_L2D', 'maml'], default='EWA')
    parser.add_argument(
        '--loss_type', choices=['softmax', 'ova'], default='softmax')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--warmstart', action='store_true')
    parser.set_defaults(warmstart=True)

    # Contextset
    parser.add_argument("--n_cntx_pts", type=int, default=50)
    parser.add_argument("--train_experts", type=int, default=3)

    # Attention
    parser.add_argument("--dim_hid", type=int, default=128)
    parser.add_argument("--depth_embed", type=int, default=6)
    parser.add_argument("--depth_reject", type=int, default=4)
    parser.add_argument('--decouple', action='store_true')
    parser.set_defaults(decouple=False)

    # Training options
    parser.add_argument("--meta_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_wrn", type=float, default=1e-2,
                        help="learning rate for wrn.")
    parser.add_argument("--lr_other", type=float, default=1e-3,
                        help="learning rate for non-wrn model components.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        '--scoring_rule', choices=['val_loss', 'sys_acc'], default='sys_acc')
    parser.add_argument("--experiment_name", type=str, default="default",
                        help="specify the experiment name. Checkpoints will be saved with this name.")
    # maml
    parser.add_argument('--n_steps_maml', type=int, default=5)
    parser.add_argument('--lr_maml', type=float, default=0.1)

    # EVAL
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument('--budget', nargs='+', type=float, default=[1.0])
    parser.add_argument('--finetune_single', action='store_true')
    parser.set_defaults(finetune_single=True)
    parser.add_argument('--n_finetune_steps', nargs='+',
                        type=int, default=[1, 2, 5, 10])
    parser.add_argument('--lr_finetune', nargs='+',
                        type=float, default=[1e-1, 1e-2])

    config = parser.parse_args().__dict__
    main(config)
