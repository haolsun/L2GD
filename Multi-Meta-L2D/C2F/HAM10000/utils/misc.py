""" Additional utility functions. """
import os
import time
import pprint
import torch
import numpy as np
import torch.nn.functional as F
import random
from itertools import combinations
import logging
import os
import copy
ROOT = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])


def ensure_path(path):
    """The function to make log path.
    Args:
      path: the generated saving path.
    """
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

class Averager():
    """The class to calculate the average."""
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc(logits, label):
    """The function to calculate the .
    Args:
      logits: input logits.
      label: ground truth labels.
    Return:
      The output accuracy.
    """
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()


class Timer():
    """The class for timer."""
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()

def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """The function to calculate the .
    Args:
      data: input records
      label: ground truth labels.
    Return:
      m: mean value
      pm: confidence interval.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def count_acc_l2d(logits, label):
    """The function to calculate the .
    Args:
      logits: input logits.
      label: ground truth labels.
    Return:
      The output accuracy.
    """
    pred = F.softmax(logits, dim=1)[:,:-1].argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()

def count_acc_sys(logits, label, exp_pred, num_class):
    """The function to calculate the .
    Args:
      logits: input logits.
      label: ground truth labels.
    Return:
      The output accuracy.
    """
    batch_size = label.shape[0]
    output = F.softmax(logits, dim=1)
    sys_correct = 0

    with torch.no_grad():
        _, prediction = torch.max(output, dim=1)
        for i in range(batch_size):
            reject = (prediction[i].item() == num_class)
            if reject:
                sys_correct += (exp_pred[i] == label[i]).item()
            else:
                sys_correct += (prediction[i] == label[i]).item()

        sys_acc = 100 * sys_correct / batch_size

    return sys_acc


def metrics_all(logits, target, exp_pred, num_class):

    output = F.softmax(logits, dim=1)

    total = 0
    exp_total = 0
    model_total = 0

    correct = 0
    exp_correct = 0
    model_correct = 0
    alone_correct = 0
    exp_alone = 0

    with torch.no_grad():

        _, model_pred = torch.max(output, dim=1)
        _, model_alone = torch.max(output[:,:num_class], dim=1)
        alone_correct += torch.eq(model_alone, target).sum()
        exp_pred = torch.tensor(exp_pred).cuda()
        exp_alone += torch.eq(exp_pred, target).sum()

        batch_size = target.shape[0]

        total += batch_size

        for i in range(batch_size):
            rej = (model_pred[i].item() == num_class)
            if rej == 1:
                exp_total += 1
                exp_correct += (exp_pred[i] == target[i]).item()
                correct += (exp_pred[i] == target[i]).item()
            elif rej == 0:
                model_total += 1
                model_correct += (model_pred[i] == target[i]).item()
                correct += (model_pred[i] == target[i]).item()

        sys_acc = 100 * correct / total
        alone_acc = (100 * alone_correct / total).item()
        exp_alone_acc = (100 * exp_alone / total).item()
        exp_acc = 100 * exp_correct / (exp_total + 0.0002)
        mod_acc = 100 * model_correct / (model_total + 0.0002)
        cov_rate = 100 * model_total / total
        if exp_total == 0:
            exp_acc = -1

        cov = str(model_total) + str("/") + str(total)
        metrics = {"cov_rate": cov_rate, "sys_acc": sys_acc, "exp_acc": exp_acc,
                   "model_acc": mod_acc, "alone_acc": alone_acc, "exp_alone":exp_alone_acc}
        # metrics_print = {"model_cov": cov, "sys_acc": round(sys_acc, 2), "exp_acc": round(exp_acc, 2),
        #            "model_acc": round(mod_acc, 2), "alone_acc": round(alone_acc, 2)}
        # print(metrics_print)

    return metrics


def count_acc_sys_multiexp(logits,labels,expert_predictions,n_classes):
    correct_sys = 0
    real_total = 0

    outputs = F.softmax(logits, dim=1)

    _, predicted = torch.max(outputs.data, 1)
    batch_size = outputs.size()[0]  # batch_size


    for i in range(0, batch_size):
        r = (predicted[i].item() >= n_classes)
        if r == 0:
            correct_sys += (predicted[i] == labels[i]).item()
        if r == 1:
            deferred_exp = (predicted[i] - n_classes).item()
            exp_prediction = expert_predictions[deferred_exp][i]
            correct_sys += (exp_prediction == labels[i].item())
        real_total += 1

    return 100 * correct_sys / real_total


def metrics_print(net,expert_fn, n_classes, loader, device):
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            logits = net(images)
            outputs = F.softmax(logits, dim=-1)
            _, predicted = torch.max(outputs.data, 1)

            batch_size = outputs.size()[0]  # batch_size
            exp_prediction = expert_fn(images, labels)
            clf_probs, clf_preds = outputs[:, :n_classes].max(dim=-1)

            for i in range(0, batch_size):
                r = (predicted[i].item() == n_classes)
                alone_correct += (clf_preds[i] == labels[i]).item()
                if r == 0:
                    total += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r == 1:
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    exp_total += 1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print = {"coverage": cov, "system_accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001),
                "alone classifier": 100 * alone_correct / real_total}
    print(to_print)
    return to_print


def multi_exp_evaluate(model,experts_list,loss_fn,n_classes,data_loader,logger=None, device=None):
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
    n_experts = len(experts_list)
    expert_correct_dic = {k: 0 for k in range(n_experts)}
    expert_total_dic = {k: 0 for k in range(n_experts)}
    #  === Individual Expert Accuracies === #

    losses = []
    exp_predictions = []
    for data in data_loader:
        if len(data) == 2:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            labels_sparse = None
        else:
            images, labels, labels_sparse = data
            images, labels, labels_sparse = images.to(device), labels.to(device), labels_sparse.to(device)

        logits = model(images)
        outputs = F.softmax(logits, dim=-1)
        clf_probs, clf_preds = outputs[:, :n_classes].max(dim=-1)
        exp_probs, defer_exps = outputs[:, n_classes:].max(dim=-1)
        _, predicted = outputs.max(dim=-1)
        is_rejection = (predicted >= n_classes).int()

        collection_Ms = []
        for idx, expert in enumerate(experts_list):
            exp_preds = torch.tensor(expert(images, labels, labels_sparse),
                                          device=device)
            costs = (exp_preds == labels).int()
            collection_Ms.append(costs)
            exp_predictions.append(exp_preds)

        loss = loss_fn(outputs, labels, collection_Ms, n_classes+n_experts)
        losses.append(loss.item())
        batch_size = len(images)
        for i in range(0, batch_size):
            r = is_rejection[i]
            prediction = clf_preds[i].item()
            defer_exp = defer_exps[i].item()
            exp_prediction = exp_predictions[defer_exp][i].item()

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
                         for k in range(n_experts)}
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

def evaluate11(model,
             expert_fns,
             loss_fn,
             all_classes,
             num_classes,
             data_loader,
             device):
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
    expert_correct_dic = {k: 0 for k in range(len(expert_fns))}
    expert_total_dic = {k: 0 for k in range(len(expert_fns))}
    #  === Individual  Expert Accuracies === #
    losses = []
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs.data, 1)
            _, defer_exp = torch.max(outputs[:, num_classes:].data, 1)

            batch_size = outputs.size()[0]  # batch_size

            expert_predictions = []
            collection_Ms = []  # a collection of 3-tuple
            for i, fn in enumerate(expert_fns, 0):
                exp_prediction1 = fn(images, labels)
                m = [0] * batch_size
                m2 = [0] * batch_size
                for j in range(0, batch_size):
                    if exp_prediction1[j] == labels[j].item():
                        m[j] = 1
                        m2[j] = 1
                    else:
                        m[j] = 0
                        m2[j] = 1

                m = torch.tensor(m)
                m2 = torch.tensor(m2)
                m = m.to(device)
                m2 = m2.to(device)
                collection_Ms.append(m)
                expert_predictions.append(exp_prediction1)

            loss = loss_fn(outputs, labels, collection_Ms, all_classes)
            losses.append(loss.item())

            for i in range(0, batch_size):
                r = (predicted[i].item() >= all_classes - len(expert_fns))
                prediction = predicted[i]
                if predicted[i] >= all_classes - len(expert_fns):
                    max_idx = 0
                    # get second max
                    for j in range(0, all_classes - len(expert_fns)):
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
                    deferred_exp = defer_exp[i].item()
                    exp_prediction = expert_predictions[deferred_exp][i]
                    #
                    # Deferral accuracy: No matter expert ===
                    exp += (exp_prediction == labels[i].item())
                    exp_total += 1
                    # Individual Expert Accuracy ===
                    expert_correct_dic[deferred_exp] += (
                        exp_prediction == labels[i].item())
                    expert_total_dic[deferred_exp] += 1
                    #
                    correct_sys += (exp_prediction == labels[i].item())
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)

    #  === Individual Expert Accuracies === #
    expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002) for k
                         in range(len(expert_fns))}
    # Add expert accuracies dict
    to_print = {"coverage": cov, "system_accuracy": 100 * correct_sys / real_total,
                "expert_accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier_accuracy": 100 * correct / (total + 0.0001),
                "alone_classifier": 100 * alone_correct / real_total,
                "validation_loss": np.average(losses),
                "n_experts": len(expert_fns),
                **expert_accuracies}
    print(to_print, flush=True)
    return to_print

def evaluate(model,
             expert_fns,
             loss_fn,
             all_classes,
             num_classes,
             data_loader,
             device):
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
    exp_alone_correct = 0
    #  === Individual Expert Accuracies === #
    expert_correct_dic = {k: 0 for k in range(len(expert_fns))}
    expert_total_dic = {k: 0 for k in range(len(expert_fns))}
    #  === Individual  Expert Accuracies === #
    losses = []
    with torch.no_grad():
        for data in data_loader:
            if len(data) == 2:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                labels_sparse = None
            else:
                images, labels, labels_sparse = data
                images, labels, labels_sparse = images.to(
                    device), labels.to(device), labels_sparse.to(device)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs.data, 1)
            _, defer_exp = torch.max(outputs[:, num_classes:].data, 1)
            _, model_pred = torch.max(outputs[:, :num_classes].data, 1)
            is_rejection = (predicted >= num_classes).int()

            batch_size = outputs.size()[0]  # batch_size

            expert_predictions = []
            collection_Ms = []  # a collection of 3-tuple
            for i, fn in enumerate(expert_fns, 0):
                exp_preds = torch.tensor(fn(images, labels, labels_sparse), device=device)
                m = (exp_preds == labels).int()
                collection_Ms.append(m)
                expert_predictions.append(exp_preds)

            loss = loss_fn(outputs, labels, collection_Ms, all_classes)
            losses.append(loss.item())

            for i in range(0, batch_size):
                r = is_rejection[i]
                prediction = model_pred[i].item()
                deferred_exp = defer_exp[i].item()
                exp_prediction = expert_predictions[deferred_exp][i].item()
                alone_correct += (prediction == labels[i]).item()
                exp_alone_correct += (exp_prediction == labels[i]).item()
                if r == 0:
                    total += 1
                    correct += (prediction == labels[i]).item()
                    correct_sys += (prediction == labels[i]).item()
                if r == 1:
                    exp += (exp_prediction == labels[i].item())
                    correct_sys += (exp_prediction == labels[i].item())
                    exp_total += 1
                    # Individual Expert Accuracy ===
                    expert_correct_dic[deferred_exp] += (
                        exp_prediction == labels[i].item())
                    expert_total_dic[deferred_exp] += 1

                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)

    #  === Individual Expert Accuracies === #
    expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002) for k
                         in range(len(expert_fns))}
    # Add expert accuracies dict
    to_print = {"coverage": cov, "system_accuracy": 100 * correct_sys / real_total,
                "expert_accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier_accuracy": 100 * correct / (total + 0.0001),
                "alone_classifier": 100 * alone_correct / real_total,
                "validation_loss": np.average(losses),
                "n_experts": len(expert_fns),
                **expert_accuracies}
    print(to_print, flush=True)
    return to_print



def evaluate_data(logits,
             expert_predictions,
             num_classes,
             data,
             device):
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
    exp_alone_correct = 0
    #  === Individual Expert Accuracies === #
    expert_correct_dic = {k: 0 for k in range(len(expert_predictions))}
    expert_total_dic = {k: 0 for k in range(len(expert_predictions))}

    #  === Individual  Expert Accuracies === #

    with torch.no_grad():
        if len(data) == 2:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            labels_sparse = None
        else:
            images, labels, labels_sparse = data
            images, labels, labels_sparse = images.to(
                device), labels.to(device), labels_sparse.to(device)

        outputs = F.softmax(logits, dim=1)

        _, predicted = torch.max(outputs.data, 1)
        _, defer_exp = torch.max(outputs[:, num_classes:].data, 1)
        _, model_pred = torch.max(outputs[:, :num_classes].data, 1)
        is_rejection = (predicted >= num_classes).int()

        batch_size = outputs.size()[0]  # batch_size

        for i in range(0, batch_size):
            r = is_rejection[i]
            prediction = model_pred[i].item()
            deferred_exp = defer_exp[i].item()
            exp_prediction = expert_predictions[deferred_exp][i].item()
            alone_correct += (prediction == labels[i]).item()
            exp_alone_correct += (exp_prediction == labels[i]).item()
            if r == 0:
                total += 1
                correct += (prediction == labels[i]).item()
                correct_sys += (prediction == labels[i]).item()
            if r == 1:
                exp += (exp_prediction == labels[i].item())
                correct_sys += (exp_prediction == labels[i].item())
                exp_total += 1
                # Individual Expert Accuracy ===
                expert_correct_dic[deferred_exp] += (
                    exp_prediction == labels[i].item())
                expert_total_dic[deferred_exp] += 1

            real_total += 1

    #  === Individual Expert Accuracies === #
    expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002) for k
                         in range(len(expert_predictions))}
    # Add expert accuracies dict
    to_print = {"system_accuracy": 100 * correct_sys / real_total,
                "expert_accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier_accuracy": 100 * correct / (total + 0.0001),
                "alone_classifier": 100 * alone_correct / real_total,
                "n_experts": len(expert_predictions),
                **expert_accuracies}
    # print(to_print, flush=True)
    return to_print

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if target.device == pred.device:
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    else:
        pred_tmp = copy.deepcopy(pred)
        pred_tmp = pred_tmp.to(target.device)
        correct = pred_tmp.eq(target.view(1, -1).expand_as(pred_tmp))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_logger(filename, mode='a'):
    try:
        os.remove(filename)
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    logger.addHandler(logging.StreamHandler())
    return logger