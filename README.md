# vision-transformers-cifar10
This is your go-to playground for training Vision Transformers (ViT) and its related models on CIFAR-10/CIFAR-100, a common benchmark dataset in computer vision.

The whole codebase is implemented in Pytorch, which makes it easier for you to tweak and experiment. Over the months, we've made several notable updates including adding different models like ConvMixer, CaiT, ViT-small, SwinTransformers, and MLP mixer. We've also adapted the default training settings for ViT to fit better with the CIFAR-10/CIFAR-100 dataset.

Using the repository is straightforward - all you need to do is run the `train_cifar10.py` script with different arguments, depending on the model and training parameters you'd like to use.

Thanks, the repo has been used in [10+ papers!](https://scholar.google.co.jp/scholar?hl=en&as_sdt=0%2C5&q=vision-transformers-cifar10&btnG=)

Please use this citation format if you use this in your research.
```
@misc{yoshioka2024visiontransformers,
  author       = {Kentaro Yoshioka},
  title        = {vision-transformers-cifar10: Training Vision Transformers (ViT) and related models on CIFAR-10},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/kentaroy47/vision-transformers-cifar10}}
}
```

### Updates
* Added [ConvMixer]((https://openreview.net/forum?id=TVHS5Y4dNvM)) implementation. Really simple! (2021/10)

* Added wandb train log to reproduce results. (2022/3)

* Added CaiT and ViT-small. (2022/3)

* Added SwinTransformers. (2022/3)

* Added MLP mixer. (2022/6)

* Changed default training settings for ViT.

* Fixed some bugs and training settings (2024/2)

* Added onnx and torchscript model exports. (2024/12)

* Added mobilevit. (2025/1)

* Add CIFAR-100 support (2025/4)

# Usage example
`python train_cifar10.py` # vit-patchsize-4

`python train_cifar10.py --dataset cifar100` # cifar-100

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

| CIFAR10 | Accuracy | Train Log |
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

| CIFAR100 | Accuracy | Train Log |
|:-----------:|:--------:|:--------:|
| ViT patch=4 Epoch@200 |    52%   | [Log](https://api.wandb.ai/links/arutema47/f8mz3mpk) |
| resnet18+randaug |    71%   | [Log](https://wandb.ai/arutema47/cifar-challenge/reports/Res18-CIFAR100--VmlldzoxMjUwNzU3Mg?accessToken=fw9ojmpfuqrrxjers2duixssezqifaonvbmf8x3ynieldw3auh53ax992g0z6cx3) |



# Used in..
* Vision Transformer Pruning [arxiv](https://arxiv.org/abs/2104.08500) [github](https://github.com/Cydia2018/ViT-cifar10-pruning)
* Understanding why ViT trains badly on small datasets: an intuitive perspective [arxiv](https://arxiv.org/abs/2302.03751)
* Training deep neural networks with adaptive momentum inspired by the quadratic optimization [arxiv](https://arxiv.org/abs/2110.09057)
* [Moderate coreset: A universal method of data selection for real-world data-efficient deep learning ](https://openreview.net/forum?id=7D5EECbOaf9)

# Model Export
This repository supports exporting trained models to ONNX and TorchScript formats for deployment purposes. You can export your trained models using the `export_models.py` script.

### Basic Usage
```bash
python export_models.py --checkpoint path/to/checkpoint --model_type vit --output_dir exported_models
