from attrdict import AttrDict
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.cifar import VisionDataset

ROOT = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])


class MyVisionDataset(VisionDataset):
    def __init__(self, images, labels, transform, labels_sparse=None):
        super().__init__(ROOT + '/data', transform=transform)
        self.data, self.targets = images, labels
        self.targets = torch.asarray(self.targets, dtype=torch.int64)
        self.targets_sparse = labels_sparse
        if self.targets_sparse is not None:
            self.targets_sparse = torch.asarray(self.targets_sparse, dtype=torch.int64)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # img = img.numpy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.targets_sparse is not None:
            target_sparse = int(self.targets_sparse[index])
            return img, target, target_sparse
        else:
            return img, target

    def __len__(self):
        return len(self.data)


# May want to extend to allow variable context set sizes
#  i.e. min_cntx_pts_per_class, max_cntx_pts_per_class
# Since batch size in data loader is fixed, just specify max value and then take subset
class ContextSampler():
    def __init__(self, images, labels, transform, labels_sparse=None, n_cntx_pts=50, device='cpu',
                 **kwargs):  # cntx_pts_per_class=5, n_classes=10
        # self.cntx_pts_per_class = cntx_pts_per_class
        self.n_cntx_pts = n_cntx_pts  # cntx_pts_per_class*n_classes
        # self.n_classes = n_classes
        self.device = device
        self.with_additional_label = False
        if labels_sparse is not None:
            self.with_additional_label = True

        # self.dataloader_lst = [] # separated by class
        # self.data_iter_lst = []
        # for cc in range(n_classes):
        #     indices = np.where(labels==cc)[0]
        #     labels_sparse_by_class = labels_sparse[indices] if self.with_additional_label else None
        #     dataset = MyVisionDataset(images[indices], labels[indices], transform, labels_sparse_by_class)
        #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=cntx_pts_per_class, shuffle=True, drop_last=True, **kwargs)
        #     self.dataloader_lst.append(dataloader)
        #     self.data_iter_lst.append(iter(dataloader))

        # Single dataloader version
        dataset = MyVisionDataset(images, labels, transform, labels_sparse)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.n_cntx_pts, shuffle=True, drop_last=True,
                                                      **kwargs)
        self.data_iter = iter(self.dataloader)

    def _balanced_sample(self):
        # input_lst = []
        # target_lst = []
        # if self.with_additional_label:
        #     target_sparse_lst = []
        # for cc in range(self.n_classes):
        #     try:
        #         data_batch = next(self.data_iter_lst[cc])
        #     except StopIteration:
        #         self.data_iter_lst[cc] = iter(self.dataloader_lst[cc])
        #         data_batch = next(self.data_iter_lst[cc])
        #     if self.with_additional_label:
        #         input, target, target_sparse = data_batch
        #         target_sparse_lst.append(target_sparse)
        #     else:
        #         input, target = data_batch
        #     input_lst.append(input)
        #     target_lst.append(target)
        # perm = torch.randperm(self.cntx_pts_per_class*self.n_classes)
        # input_all = torch.vstack(input_lst)[perm]
        # target_all = torch.cat(target_lst)[perm]
        # if self.with_additional_label:
        #     target_sparse_all = torch.cat(target_sparse_lst)[perm]
        #     input_all, target_all, target_sparse_all = input_all.to(self.device), target_all.to(self.device), target_sparse_all.to(self.device)
        #     return input_all, target_all, target_sparse_all
        # else:
        #     input_all, target_all = input_all.to(self.device), target_all.to(self.device)
        #     return input_all, target_all

        # Single dataloader version
        try:
            data_batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            data_batch = next(self.data_iter)

        if self.with_additional_label:
            input_all, target_all, target_all_sparse = data_batch
            input_all, target_all, target_all_sparse = input_all.to(self.device), target_all.to(
                self.device), target_all_sparse.to(self.device)
            return input_all, target_all, target_all_sparse
        else:
            input_all, target_all = data_batch
            input_all, target_all = input_all.to(self.device), target_all.to(self.device)
            return input_all, target_all

    def sample(self, n_experts=1):
        # input_lst = []
        # target_lst = []
        # for _ in range(n_experts):
        #     input, target = self._balanced_sample()
        #     input_lst.append(input.unsqueeze(0))
        #     target_lst.append(target.unsqueeze(0))
        # cntx = AttrDict()
        # cntx.xc = torch.vstack(input_lst)
        # cntx.yc = torch.vstack(target_lst)

        # Not resample for multiple experts (at train-time)
        cntx = AttrDict()
        if self.with_additional_label:
            input, target, target_sparse = self._balanced_sample()
            cntx.yc_sparse = target_sparse.unsqueeze(0).repeat(n_experts, 1)
        else:
            input, target = self._balanced_sample()
            cntx.yc_sparse = None
        cntx.xc = input.unsqueeze(0).repeat(n_experts, 1, 1, 1, 1)
        cntx.yc = target.unsqueeze(0).repeat(n_experts, 1)

        return cntx

    def reset(self):
        # for cc in range(self.n_classes):
        #     self.data_iter_lst[cc] = iter(self.dataloader_lst[cc])

        self.data_iter = iter(self.dataloader)


def load_gtsrb():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [87.1, 79.7, 82.0]],
                                     std=[x / 255.0 for x in [69.8, 66.5, 67.9]])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transform_train

    train_dataset_all = datasets.GTSRB(root=ROOT + '/data', split='train', download=True)
    transform_resize = transforms.Resize((32, 32))  # Resize all images to 32x32 (originals are variable size)

    train_images_all = np.vstack(
        [np.array(transform_resize(train_dataset_all[i][0]))[None, :] for i in range(len(train_dataset_all))])
    train_targets_all = [train_dataset_all[i][1] for i in range(len(train_dataset_all))]

    test_dataset = datasets.GTSRB(root=ROOT + '/data', split='test', download=True)
    test_images_all = np.vstack(
        [np.array(transform_resize(test_dataset[i][0]))[None, :] for i in range(len(test_dataset))])
    test_targets_all = [test_dataset[i][1] for i in range(len(test_dataset))]

    # images_all = np.vstack((train_images_all, test_images_all))
    # targets_all = train_targets_all + test_targets_all

    # Extract 10,000 examples from full train set + seeded
    images_train, _, targets_train, _ = \
        train_test_split(train_images_all, train_targets_all, train_size=10000, random_state=0,
                         stratify=train_targets_all)

    # 50/50 split into val/test (unseeded)
    images_val, images_test, targets_val, targets_test = \
        train_test_split(test_images_all, test_targets_all, train_size=0.5, random_state=0, stratify=test_targets_all)

    train_dataset = MyVisionDataset(images_train, targets_train, transform_train)
    val_dataset = MyVisionDataset(images_val, targets_val, transform_test)
    test_dataset = MyVisionDataset(images_test, targets_test, transform_test)

    return train_dataset, val_dataset, test_dataset
