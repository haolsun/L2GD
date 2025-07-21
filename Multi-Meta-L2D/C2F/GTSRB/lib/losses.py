import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F


def reject_CrossEntropyLoss(outputs, labels, m, n_classes=10):
    '''
    The L_{CE} loss implementation for CIFAR with alpha=1
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (I_{m =y})
    labels: target
    n_classes: number of classes
    '''
    outputs = F.softmax(outputs, dim=-1)
    batch_size = outputs.size()[0]
    rc = [n_classes] * batch_size # idx to extract rejector function
    a = -m * torch.log2(outputs[range(batch_size), rc])
    b = - torch.log2(outputs[range(batch_size), labels])
    outputs = a + b
    return torch.sum(outputs) / batch_size

def multi_reject_CrossEntropyLoss(outputs, labels, collection_Ms, n_classes, with_softmax=True):
    '''
    The L_{CE} loss implementation for CIFAR
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    labels: target
    m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    n_classes: number of classes
    '''
    if with_softmax:
        outputs = F.softmax(outputs, dim=-1)
    batch_size = outputs.size()[0]  # batch_size
    rcs = []
    for i, _ in enumerate(collection_Ms, 0):
        rcs.append([n_classes-(i+1)] * batch_size)

    temp = -torch.log2(outputs[range(batch_size), labels])
    for i, m in enumerate(collection_Ms):
        temp -= m * \
            torch.log2(outputs[range(batch_size), rcs[len(rcs)-i-1]])
    return torch.sum(temp) / batch_size


def softmax(outputs, labels, collection_Ms, n_classes):
    '''
    The L_{CE} loss implementation for CIFAR
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    labels: target
    m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    n_classes: number of classes
    '''
    batch_size = outputs.size()[0]  # batch_size
    m2 = collection_Ms[0][1]

    rcs = []
    for i, _ in enumerate(collection_Ms, 0):
        rcs.append([n_classes-(i+1)] * batch_size)

    temp = -m2 * torch.log2(outputs[range(batch_size), labels])
    for i, (m, _) in enumerate(collection_Ms):
        temp -= m * \
            torch.log2(outputs[range(batch_size), rcs[len(rcs)-i-1]])
    return torch.sum(temp) / batch_size
