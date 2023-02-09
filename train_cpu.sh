#/bin/bash

wandb off
python train_cifar10.py --net swin --n_epochs 400 --noamp --device cpu
