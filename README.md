# KEEN: Hyperbolic Knowledge graph Embedding in Extended Poincaré ball

This repository contains our implementation of paper **Hyperbolic Knowledge Graph Embedding in Extended Poincaré Ball**. *Xingchen Zhou, Peng Wang, Zhe Pan.*

## 1. Dependencies

* Python 3.6+
* PyTorch 1.0+
* numpy 1.14+

## 2. Results

The results of **KEEN** on **WN18RR**, **FB15k-237** and **YAGO3-10** are shown as follows.

| Datasets  | MRR  | H@1  | H@3  | H@10 |
| --------- | ---- | ---- | ---- | ---- |
| WN18RR    | .488 | .448 | .502 | .570 |
| FB15k-237 | .340 | .243 | .379 | .541 |
| YAGO3-10  | .499 | .411 | .553 | .667 |

## 3. Instructions for running

To analyze **KEEN**'s performance on the datasets, please run our code as follows:

### 3.1 WN18RR

```python
CUDA_VISIBLE_DEVICES=0 python -u runs.py --do_train --do_valid --do_test --data_path ./data/wn18rr/ --model KEEN -n 1024 -b 256 -d 512 -g 6.0 -a 0.5 -lr 0.0001 --max_steps 80000 -save models/KEEN_wn18rr --test_batch_size 8 --cuda
```

### 3.2 FB15k-237

```python
CUDA_VISIBLE_DEVICES=1 python -u codes/runs.py --do_train --do_valid --do_test --data_path ./data/FB15k-237/ --model KEEN -n 1024 -b 256 -d 1000 -g 9.0 -a 1.0 -lr 0.0001 --max_steps 80000 -save models/KEEN_FB15k-237 --test_batch_size 8 --cuda
```

### 3.3 YAGO3-10

```python
CUDA_VISIBLE_DEVICES=2 python -u codes/runs.py --do_train --do_valid --do_test --data_path ./data/YAGO3-10/ --model KEEN -n 512 -b 256 -d 512 -g 24.0 -a 1.0 -lr 0.002 --max_steps 180000 -save models/KEEN_YAGO3-10 --test_batch_size 8 --cuda
```

## 4. Acknowledgement

We refer to the code of [KGE-HAKE](https://github.com/MIRALab-USTC/KGE-HAKE). Thanks for their contributions.

## 5. Licences

Every source code file written exclusively by the author of this repo is licensed under Apache License Version 2.0. For more information, please refer to the [license](https://github.com/prokolyvakis/hyperkg/blob/master/LICENSE).