# MedMNIST L2D
This program is about l2d algorithm on the mnist dataset
## WarmStart
If you want to use a pre trained feature extractor, please run the `train_classifier.py` file first. And set the `warmstart` parameter when training the model with `main.py`
```
python run_classifier.py --dataset=organamnist 
```
## MedMNIST Experiment
If you want to implement domain generalization experiments from **OrganANMNIST to OrganANMNIST**, please execute the following command:
```python
python main.py --mode=train --dataset=organamnist --test_dataset=organcmnist --gpu_ids=0 --l2d=EWA --train_experts=3 --warmstart
```
Here are explanations of some important args,
```
--l2d: specifies l2d algorithm, including 6 types, namely Multi,MAML,EWA,EWA_attn,pop,pop_attn
```

### eval
When you train a model on a dataset (such as OrganANMNIST), you can perform domain generalization testing directly on three datasets by executing the following command:
```
python main.py --mode=eval --dataset=organamnist --test_dataset=organsmnist --gpu_ids=0 --l2d=EWA --train_experts=3 --time_stamp={train_time}
```
- `{train_time}` is the last level directory where the trained model is stored, with a format similar to `%y%m%d_%H%M%S`

## Adaptive Number of Experts
Only `EWA` and `EWA_pop` algorithms can achieve this experiment by executing the following command:
```
python main_expert_num_vary.py --mode=eval --l2d=EWA --train_experts=3 --test_experts=4 time_stamp={train_time} --gpu_ids=0
```
