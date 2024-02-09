# vision-transformers-cifar10
This is your go-to playground for training Vision Transformers (ViT) and its related models on CIFAR-10, a common benchmark dataset in computer vision.

The whole codebase is implemented in Pytorch, which makes it easier for you to tweak and experiment. Over the months, we've made several notable updates including adding different models like ConvMixer, CaiT, ViT-small, SwinTransformers, and MLP mixer. We've also adapted the default training settings for ViT to fit better with the CIFAR-10 dataset.

Using the repository is straightforward - all you need to do is run the `train_cifar10.py` script with different arguments, depending on the model and training parameters you'd like to use.

### Updates
* Added [ConvMixer]((https://openreview.net/forum?id=TVHS5Y4dNvM)) implementation. Really simple! (2021/10)

* Added wandb train log to reproduce results. (2022/3)

* Added CaiT and ViT-small. (2022/3)

* Added SwinTransformers. (2022/3)

* Added MLP mixer. (2022/6)

* Changed default training settings for ViT.

* Fixed some bugs and training settings (2024/2)

# Usage example
`python train_cifar10.py` # vit-patchsize-4

`python train_cifar10.py  --size 48` # vit-patchsize-4-imsize-48

`python train_cifar10.py --patch 2` # vit-patchsize-2

`python train_cifar10.py --net vit_small --n_epochs 400` # vit-small

`python train_cifar10.py --net vit_timm` # train with pretrained vit

`python train_cifar10.py --net convmixer --n_epochs 400` # train with convmixer

`python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3`

`python train_cifar10.py --net cait --n_epochs 200` # train with cait

`python train_cifar10.py --net swin --n_epochs 400` # train with SwinTransformers

`python train_cifar10.py --net res18` # resnet18+randaug

# Results..

|             | Accuracy | Train Log |
|:-----------:|:--------:|:--------:|
| ViT patch=2 |    80%    | |
| ViT patch=4 Epoch@200 |    80%   | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTU2?accessToken=3y3ib62e8b9ed2m2zb22dze8955fwuhljl5l4po1d5a3u9b7yzek1tz7a0d4i57r) |
| ViT patch=4 Epoch@500 |    88%   | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTU2?accessToken=3y3ib62e8b9ed2m2zb22dze8955fwuhljl5l4po1d5a3u9b7yzek1tz7a0d4i57r) |
| ViT patch=8 |    30%   | |
| ViT small  | 80% | |
| MLP mixer |    88%   | |
| CaiT  | 80% | |
| Swin-t  | 90% | |
| ViT small (timm transfer) | 97.5% | |
| ViT base (timm transfer) | 98.5% | |
| [ConvMixerTiny(no pretrain)](https://openreview.net/forum?id=TVHS5Y4dNvM) | 96.3% |[Log](https://wandb.ai/arutema47/cifar10-challange/reports/convmixer--VmlldzoyMjEyOTk1?accessToken=2w9nox10so11ixf7t0imdhxq1rf1ftgzyax4r9h896iekm2byfifz3b7hkv3klrt)|
|   resnet18  |  93%  | |
|   resnet18+randaug  |  95%  | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTYz?accessToken=968duvoqt6xq7ep75ob0yppkzbxd0q03gxy2apytryv04a84xvj8ysdfvdaakij2) |

# Used in..
* Vision Transformer Pruning [arxiv](https://arxiv.org/abs/2104.08500) [github](https://github.com/Cydia2018/ViT-cifar10-pruning)
