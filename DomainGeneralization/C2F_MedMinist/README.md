# C2F L2D
This program is about l2d-C2F algorithm 
## MedMNIST Dataset
You can refer to this repository for the use of the MedMNIST dataset:<https://github.com/MedMNIST/MedMNIST>
### Pre-training
The C2F algorithm is divided into two stages, pre-training and meta-learning. Pre-train using the following command
```
python main.py --dataset=organamnist --phase=pre_train --pre_experts=8 --pre_lr=0.01
```
### Meta-training
```
python main.py --dataset=organamnist --phase=meta_train --meta_experts=3 --time_stamp_pre={pre_train_time}
```
- `{pre_train_time}` is the last level directory where the pretrained model is stored, with a format similar to `%y%m%d_%H%M%S`
### Domain generalization experiment
The domain generalization experiment is performed using the following command
```
python main.py --dataset=organamnist --test_dataset=organcmnist --phase=meta_eval --time_stamp_pre={pre_train_time} --time_stamp_meta={pre_train_meta}
```
- `{meta_train_time}` is the last level directory where the meta-traine model is stored, with a format similar to `%y%m%d_%H%M%S`
