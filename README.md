# vision-transformers-cifar10
Let's train vision transformers for cifar 10! 

This is an unofficial and elementary implementation of `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`.

I use pytorch for implementation.

# Usage
`python train_cifar10.py` # vit-patchsize-4

`python train_cifar10.py --patch 2` # vit-patchsize-2

`python train_cifar10.py --net res18` # resnet18

`python train_cifar10.py --net vit_timm --lr 1e-4` # train with pretrained vit

# Results..

|             | Accuracy |
|:-----------:|:--------:|
| ViT patch=2 |    80%    |
| ViT patch=4 |    80%   |
| ViT patch=8 |    30%   |
| ViT small (timm transfer) | 97.5% |
| ViT base (timm transfer) | 98.5% |
| ViT large (timm transfer) | ??% |
|   resnet18  |  93%  |

# Used in..
* Vision Transformer Pruning [arxiv](https://arxiv.org/abs/2104.08500) [github](https://github.com/Cydia2018/ViT-cifar10-pruning)
