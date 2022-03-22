# vision-transformers-cifar10
Let's train vision transformers for cifar 10! 

This is an unofficial and elementary implementation of `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`.

I use pytorch for implementation.

### Updates
* Added [ConvMixer]((https://openreview.net/forum?id=TVHS5Y4dNvM)) implementation. Really simple! (2021/10)

* Added wandb train log to reproduce results. (2022/3)

* Added CaiT and ViT-small. (2022/3)

* Added SwinTransformers. (2022/3)


# Usage example
`python train_cifar10.py --lr 1e-4  --aug --n_epochs 200` # vit-patchsize-4

`python train_cifar10.py --lr 1e-4  --aug --n_epochs 200 --size 48` # vit-patchsize-4-imsize-48

`python train_cifar10.py --patch 2  --lr 1e-4  --aug --n_epochs 200` # vit-patchsize-2

`python train_cifar10.py --net vit_small --lr 1e-4  --aug --n_epochs 200` # vit-small

`python train_cifar10.py --net vit_timm --lr 1e-4` # train with pretrained vit

`python train_cifar10.py --net convmixer --aug --n_epochs 200` # train with convmixer

`python train_cifar10.py --net cait --lr 1e-4  --aug --n_epochs 200` # train with cait

`python train_cifar10.py --net swin --lr 1e-4  --aug --n_epochs 200` # train with SwinTransformers

`python train_cifar10.py --net res18` # resnet18

`python train_cifar10.py --net res18 --aug --n_epochs 200` # resnet18+randaug

# Results..

|             | Accuracy | Train Log |
|:-----------:|:--------:|:--------:|
| ViT patch=2 |    80%    | |
| ViT patch=4 Epoch@200 |    80%   | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTU2?accessToken=3y3ib62e8b9ed2m2zb22dze8955fwuhljl5l4po1d5a3u9b7yzek1tz7a0d4i57r) |
| ViT patch=4 Epoch@500 |    88%   | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTU2?accessToken=3y3ib62e8b9ed2m2zb22dze8955fwuhljl5l4po1d5a3u9b7yzek1tz7a0d4i57r) |
| ViT patch=8 |    30%   | |
| ViT small  | 80% | |
| CaiT  | 80% | |
| Swin-t  | 90% | |
| ViT small (timm transfer) | 97.5% | |
| ViT base (timm transfer) | 98.5% | |
| [ConvMixerTiny(no pretrain)](https://openreview.net/forum?id=TVHS5Y4dNvM) | 96.3% | |
|   resnet18  |  93%  | |
|   resnet18+randaug  |  95%  | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTYz?accessToken=968duvoqt6xq7ep75ob0yppkzbxd0q03gxy2apytryv04a84xvj8ysdfvdaakij2) |

# Used in..
* Vision Transformer Pruning [arxiv](https://arxiv.org/abs/2104.08500) [github](https://github.com/Cydia2018/ViT-cifar10-pruning)
