import random
import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# local imports
from lib.utils import AverageMeter, accuracy, get_logger
from lib.modules import Classifier

import medmnist
import numpy as np
import PIL
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import INFO
from lib.models import ResNet18, ResNet50
from torchvision.models import resnet18, resnet50

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate(model,
            n_classes,
            data_loader,
            config,
            logger):
    '''
    Computes metrics for deferal
    -----
    Arguments:
    net: model
    expert_fn: expert model
    n_classes: number of classes
    loader: data loader
    '''
    real_total = 0
    clf_alone_correct = 0
    losses = []
    model.eval() # Crucial for networks with batchnorm layers!
    criterion = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        for data in data_loader:
            if len(data) == 2:
                images, labels = data
            else:
                images, labels, _ = data # ignore additional labels

            images = images.to(device)

            if config["task"] == 'multi-label, binary-class':
                labels = labels.to(torch.float32).to(device)
            else:
                labels = torch.squeeze(labels, 1).long().to(device)

            batch_size = len(images)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, -1)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            for i in range(0, batch_size):
                prediction = predicted[i]
                clf_alone_correct += (prediction == labels[i]).item()
                real_total += 1
    metrics = {"clf_acc": 100 * clf_alone_correct / real_total,
                "val_loss": np.average(losses)}
    to_print = ""
    for k,v in metrics.items():
        if type(v)==str:
            to_print += f"{k} {v} "
        else:
            to_print += f"{k} {v:.6f} "
    logger.info(to_print)
    return metrics


def train_epoch(iters,
                train_loader,
                model,
                optimizer,
                scheduler,
                epoch,
                n_classes,
                config,
                logger):
    """ Train for one epoch """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    epoch_train_loss = []

    criterion = nn.CrossEntropyLoss(reduction='mean')
    for i, data in enumerate(train_loader):
        if len(data) == 2:
            input, target = data
        else:
            input, target, _ = data # ignore additional labels
        input = input.to(device)

        if config["task"] == 'multi-label, binary-class':
            target = target.to(torch.float32).to(device)
        else:
            target = torch.squeeze(target, 1).long().to(device)

        outputs = model(input) # [B,K]
        loss = criterion(outputs, target)
        epoch_train_loss.append(loss.item())

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, target, topk=(1,))[0] # just measures clf accuracy
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        iters+=1

        # if i % 10 == 0:
        if i % 50 == 0:
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
          config):
    logger = get_logger(os.path.join(config["ckp_dir"], "train.log"))
    logger.info(f"seed={config['seed']}")
    logger.info(config)
    logger.info('No. of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    n_classes = config["n_classes"]
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["train_batch_size"], shuffle=True, **kwargs) # drop_last=True
    valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=config["train_batch_size"], shuffle=False, **kwargs) # shuffle=True, drop_last=True

    model = model.to(device)
    cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(), config["lr"], momentum=0.9, nesterov=True, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * config["epochs"])

    iters = 0
    lrate = config["lr"]

    for epoch in range(0, config["epochs"]):
        iters, train_loss = train_epoch(iters, 
                                        train_loader, 
                                        model, 
                                        optimizer, 
                                        scheduler, 
                                        epoch,
                                        n_classes,
                                        config,
                                        logger)
        metrics = evaluate(model,
                           n_classes,
                           valid_loader,
                           config,
                           logger)

        validation_loss = metrics["val_loss"]



def eval(model, test_data, config):

    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["test_batch_size"], shuffle=False, **kwargs)
    logger = get_logger(os.path.join(config["ckp_dir"], "eval.log"))
    evaluate(model, config["n_classes"], test_loader, config, logger)


def main(config):
    set_seed(config["seed"])

    dataset = config['dataset']
    download = config['download']
    as_rgb = config["as_rgb"]
    size = config["size"]
    resize = config["resize"]
    model_flag = config["model_flag"]

    info = INFO[dataset]
    config["task"] = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])
    config["n_classes"] = n_classes
    config["epochs"] = 100
    config["lr"] = 0.1

    config["ckp_dir"] = f"./pretrained/{config['dataset']}/seed{str(config['seed'])}"
    os.makedirs(config["ckp_dir"], exist_ok=True)

    if model_flag == 'resnet18':
        resnet_base = resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels,
                                                                                                num_classes=n_classes,
                                                                                                norm_type=config[
                                                                                                    "norm_type"])
    elif model_flag == 'resnet50':
        resnet_base = resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels,
                                                                                                num_classes=n_classes,
                                                                                                norm_type=config[
                                                                                                    "norm_type"])
    else:
        raise NotImplementedError

    n_features = resnet_base.n_features

    model = Classifier(resnet_base, num_classes=int(config["n_classes"]), n_features=n_features, with_softmax=False)

    DataClass = getattr(medmnist, info['python_class'])

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

    train_data = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb, size=size)
    val_data = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb, size=size)
    test_data = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb, size=size)

    
    train(model, train_data, val_data, config)
    eval(model, test_data, config)

    # save only resnet part
    torch.save(resnet_base.state_dict(), os.path.join(config["ckp_dir"], config["experiment_name"] + ".pt"))
    with open(os.path.join(config["ckp_dir"], config["experiment_name"] + ".json"), "w") as f:
        json.dump(config, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--experiment_name", type=str, default="default",
                            help="specify the experiment name. Checkpoints will be saved with this name.")
    parser.add_argument('--dataset', type=str, default='organamnist',
                        choices=["organamnist", "organcmnist", "organsmnist"])
    parser.add_argument("--norm_type", choices=["batchnorm", "frn"], default="frn") # only for cifar/wrn

    ## MedMNIST2D
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
    
    config = parser.parse_args().__dict__
    main(config)


def doc():
    '''
    PneumoniaMNIST
    clf_acc 85.737179 val_loss 0.923104

    OrganAMNIST
    clf_acc 91.669479 val_loss 0.387427

    OrganCMNIST
    clf_acc 90.481986 val_loss 0.424089

    OrganSMNIST
    clf_acc 76.639855 val_loss 0.998454



    :return:
    '''