# vision-transformers-cifar10
Let's train vision transformers for cifar 10! 

This is an unofficial and elementary implementation of `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`.

I use pytorch for implementation.

### Updates
Added [ConvMixer]((https://openreview.net/forum?id=TVHS5Y4dNvM)) implementation. Really simple! (2021/10)


# Usage
`python train_cifar10.py` # vit-patchsize-4

`python train_cifar10.py --patch 2` # vit-patchsize-2

`python train_cifar10.py --net vit_timm --lr 1e-4` # train with pretrained vit

`python train_cifar10.py --net convmixer --aug --n_epochs 200` # train with convmixer

`python train_cifar10.py --net res18` # resnet18

`python train_cifar10.py --net res18 --aug --n_epochs 200` # resnet18+randaug

# Results..

|             | Accuracy |
|:-----------:|:--------:|
| ViT patch=2 |    80%    |
| ViT patch=4 |    80%   |
| ViT patch=8 |    30%   |
| ViT small (timm transfer) | 97.5% |
| ViT base (timm transfer) | 98.5% |
| [ConvMixerTiny(no pretrain)](https://openreview.net/forum?id=TVHS5Y4dNvM) | 96.3% |
|   resnet18  |  93%  |
|   resnet18+randaug  |  95%  |

# Used in..
* Vision Transformer Pruning [arxiv](https://arxiv.org/abs/2104.08500) [github](https://github.com/Cydia2018/ViT-cifar10-pruning)
