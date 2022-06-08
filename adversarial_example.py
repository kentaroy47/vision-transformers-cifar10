# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from collections import OrderedDict

from models import *
from models.vit import ViT
from utils import progress_bar
from models.convmixer import ConvMixer, ConvMixer_adv
from randomaug import RandAugment

from autoattack import AutoAttack # adversarial example


# transformation
class pixel_shuffling(object):
    def __init__(self):
        pass
    def __call__(self, x):
        M=4   # block size
        result_x = torch.zeros_like(x)
        C, H, W = x.shape
        # torch.manual_seed(0)
        # key = torch.randperm(M*M*C)
        key = torch.tensor([
            44, 22, 29, 18, 23, 20, 19, 41, 25, 39,  4,  2, 36, 32,  1, 13, 40, 10,
            15, 33, 26,  3, 31, 27, 21, 30, 35,  0, 37, 38, 14, 12,  5,  6, 28, 43,
            46, 45,  9,  8, 42,  7, 11, 47, 24, 17, 34, 16
        ])
        for h in range(0, H, M):
            for w in range(0, W, M):
                tmp = torch.ravel(x[:,h:h+M,w:w+M])
                tmp = tmp[key]
                tmp = torch.reshape(tmp, (C,M,M))
                result_x[:,h:h+M,w:w+M] = tmp
        return result_x

class block_scrambling(object):
    def __init__(self):
        pass
    def __call__(self, x):
        M=4   # block size
        C, H, W = x.shape
        result_x = torch.zeros_like(x)
        # torch.manual_seed(0)
        # key = torch.randperm(int((H/M)*(W/M)))
        key = torch.tensor([
            44, 46, 17,  3, 47, 21, 35,  6, 33,  2, 63, 19, 28, 22, 42, 11, 40,  4,
            14, 13, 15, 52,  8, 45, 48, 60, 55, 16, 61, 54,  9,  1, 51, 32, 59, 49,
            31, 10, 26,  5, 18,  0, 62, 27, 38, 50, 34, 41, 43, 23, 56, 25, 57, 37,
            30, 20, 53, 12, 29, 39,  7, 24, 58, 36
        ])
        blocks = torch.zeros((int((H/M)*(W/M)),C,M,M))
        for h in range(0,H,M):
          for w in range(0,W,M):
            blocks[int(h/M)*int(W/M)+int(w/M),:,:,:] = x[:,h:h+M,w:w+M]
        blocks = blocks[key]
        for h in range(0,H,M):
            for w in range(0,W,M):
                result_x[:,h:h+M, w:w+M] = blocks[int(h/M)*int(W/M) + int(w/M)]
        return result_x

class bit_flipping(object):
    def __init__(self):
        pass
    def __call__(self, x):
        # print(x[0, 0, :5]) 0<=x<=1
        M=4   # block size
        result_x = torch.zeros_like(x)
        C, H, W = x.shape
        # torch.manual_seed(0)
        # key = torch.tensor([i%2 for i in range(M*M*C)])
        # key = key[torch.randperm(M*M*C)]
        key = torch.tensor([
            0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
            1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0
        ])
        for h in range(0, H, M):
            for w in range(0, W, M):
                tmp = torch.ravel(x[:,h:h+M,w:w+M])
                tmp[key==1] = 1. - tmp[key==1]
                tmp = torch.reshape(tmp, (C,M,M))
                result_x[:,h:h+M,w:w+M] = tmp
        return result_x

#####
class pixel_shuffling_new(object):
    def __init__(self, seed, M=4, C=3):
        torch.manual_seed(seed=seed)
        self.key = torch.randperm(M*M*C)
    def __call__(self, x):
        M=4   # block size
        result_x = torch.zeros_like(x)
        C, H, W = x.shape
        for h in range(0, H, M):
            for w in range(0, W, M):
                tmp = torch.ravel(x[:,h:h+M,w:w+M])
                tmp = tmp[self.key]
                tmp = torch.reshape(tmp, (C,M,M))
                result_x[:,h:h+M,w:w+M] = tmp
        return result_x

class block_scrambling_new(object):
    def __init__(self, seed, M=4, C=3, H=32, W=32):
        torch.manual_seed(seed)
        self.key = torch.randperm(int((H/M)*(W/M)))
    def __call__(self, x):
        M=4   # block size
        C, H, W = x.shape
        result_x = torch.zeros_like(x)
        blocks = torch.zeros((int((H/M)*(W/M)),C,M,M))
        for h in range(0,H,M):
          for w in range(0,W,M):
            blocks[int(h/M)*int(W/M)+int(w/M),:,:,:] = x[:,h:h+M,w:w+M]
        blocks = blocks[self.key]
        for h in range(0,H,M):
            for w in range(0,W,M):
                result_x[:,h:h+M, w:w+M] = blocks[int(h/M)*int(W/M) + int(w/M)]
        return result_x

class bit_flipping_new(object):
    def __init__(self, seed, M=4, C=3):
        torch.manual_seed(seed=seed)
        self.key = torch.tensor([i%2 for i in range(M*M*C)])
        self.key = self.key[torch.randperm(M*M*C)]
    def __call__(self, x):
        M=4   # block size
        result_x = torch.zeros_like(x)
        C, H, W = x.shape
        for h in range(0, H, M):
            for w in range(0, W, M):
                tmp = torch.ravel(x[:,h:h+M,w:w+M])
                tmp[self.key==1] = 1. - tmp[self.key==1]
                tmp = torch.reshape(tmp, (C,M,M))
                result_x[:,h:h+M,w:w+M] = tmp
        return result_x

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='128')
parser.add_argument('--n_epochs', type=int, default='50')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--convkernel', default='8', type=int)
parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')

args = parser.parse_args()

# take in args
import wandb
watermark = "{}_lr{}".format(args.net, args.lr)
if args.amp:
    watermark += "_useamp"

wandb.init(project="cifar10-challange",
           name=watermark)
wandb.config.update(args)

if args.aug:
    import albumentations
bs = int(args.bs)

use_amp = args.amp

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = 32
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    # block_scrambling(),
    pixel_shuffling_new(seed=0),
    bit_flipping_new(seed=0),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # pixel_shuffling(),
])

# Add RandAugment with N, M(hyperparameter)
if args.aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

trainset = torchvision.datasets.CIFAR10(root='../../../data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='../../../data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    # net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10) # 
    net = ConvMixer(256, 8, kernel_size=9, patch_size=4, n_classes=10) # for p4dataall
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/convmixer-4-ckpt3.pth', map_location=torch.device('cuda:0'))
    checkpoint = torch.load('./checkpoint/convmixer-p4-epoch200-data-plain-ckpt.pth', map_location=torch.device('cuda:0')) # original model
    new_checkpoint = {}
    for key in checkpoint.keys():
        new_checkpoint[key[7:]] = checkpoint[key]

    # # pixel shuffling
    # print(new_checkpoint['0.weight'].shape)
    
    # M = 4 # block size = patch size
    # C = 3 # num of channels
    # torch.manual_seed(seed=0) # 0 is correct key
    # key = torch.randperm(M*M*C)
    # dim, C, H, W = new_checkpoint['0.weight'].shape
    # tmp = new_checkpoint['0.weight'].reshape((dim, C*H*W))
    # tmp = tmp[:, key]
    # new_checkpoint['0.weight'] = tmp.reshape((dim, C, H, W))

    # # bit flipping
    # print(new_checkpoint['0.weight'].shape)
    # M = 4 # block size = patch size
    # C = 3 # num of channels
    # torch.manual_seed(seed=0) # 0 is correct key
    # key = torch.tensor([i%2 for i in range(M*M*C)])
    # key = key[torch.randperm(M*M*C)]
    # dim, C, H, W = new_checkpoint['0.weight'].shape
    # tmp = new_checkpoint['0.weight'].reshape((dim, C*H*W))
    # tmp[:, key==1] *= -1
    # new_checkpoint['0.weight'] = tmp.reshape((dim, C, H, W))

    net.load_state_dict(new_checkpoint)
    # for param in net.parameters():
    #     print(param.shape, param.names)
elif args.net=="convmixer_adv":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    # net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10) # 
    net = ConvMixer_adv(256, 8, kernel_size=9, patch_size=4, n_classes=10) # for p4dataall
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/convmixer-4-ckpt3.pth', map_location=torch.device('cuda:0'))
    checkpoint = torch.load('./checkpoint/convmixer_adv-p4-epoch200-data-plain-ckpt.pth', map_location=torch.device('cuda:0')) # original model
    # checkpoint = torch.load('./checkpoint/convmixer2-p4-epoch200-data-all-ckpt.pth', map_location=torch.device('cuda:2')) # with postion_embeddings
    new_checkpoint = {}
    for key in checkpoint.keys():
        # print(key)
        new_checkpoint[key[7:]] = checkpoint[key]

    # pixel shuffling
    print(new_checkpoint['patch_embedding.weight'].shape)
    
    M = 4 # block size = patch size
    C = 3 # num of channels
    torch.manual_seed(seed=0) # 0 is correct key
    key = torch.randperm(M*M*C)
    dim, C, H, W = new_checkpoint['patch_embedding.weight'].shape
    tmp = new_checkpoint['patch_embedding.weight'].reshape((dim, C*H*W))
    tmp = tmp[:, key]
    new_checkpoint['patch_embedding.weight'] = tmp.reshape((dim, C, H, W))

    # bit flipping
    print(new_checkpoint['patch_embedding.weight'].shape)
    M = 4 # block size = patch size
    C = 3 # num of channels
    torch.manual_seed(seed=0) # 0 is correct key
    key = torch.tensor([i%2 for i in range(M*M*C)])
    key = key[torch.randperm(M*M*C)]
    dim, C, H, W = new_checkpoint['patch_embedding.weight'].shape
    tmp = new_checkpoint['patch_embedding.weight'].reshape((dim, C*H*W))
    tmp[:, key==1] *= -1
    new_checkpoint['patch_embedding.weight'] = tmp.reshape((dim, C, H, W))

    net.load_state_dict(new_checkpoint) 
    # for param in net.parameters():
    #     print(param.shape, param.names)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = 32,
    patch_size = args.patch,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_large_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)

net = net.to(device)
if device == 'cuda:0':
    net = torch.nn.DataParallel(net, device_ids=[0]) # make parallel
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine or reduce LR on Plateau scheduling
if not args.cos:
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

if args.cos:
    wandb.config.scheduler = "cosine"
else:
    wandb.config.scheduler = "ReduceLROnPlateau"

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    print(acc)
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {"model": net.state_dict(),
    #           "optimizer": optimizer.state_dict(),
    #           "scaler": scaler.state_dict()}
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
    #     best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

# adversarial test
def adversarial_test(epoch, attack_type='apgd-ce'): # attack type: ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    adversary = AutoAttack(net, norm='Linf', eps=8/255, version='standard') #
    adversary.attacks_to_run = [attack_type]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            dict_adv = adversary.run_standard_evaluation_individual(inputs, targets, bs=bs) #

            outputs = net(dict_adv[attack_type])
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    print(acc)
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {"model": net.state_dict(),
    #           "optimizer": optimizer.state_dict(),
    #           "scaler": scaler.state_dict()}
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
    #     best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

# list_loss = []
# list_acc = []

wandb.watch(net)
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    # trainloss = train(epoch)
    val_loss, acc = test(epoch)
    val_loss, acc = adversarial_test(epoch, attack_type='apgd-ce')
    val_loss, acc = adversarial_test(epoch, attack_type='apgd-t')
    val_loss, acc = adversarial_test(epoch, attack_type='fab-t')
    val_loss, acc = adversarial_test(epoch, attack_type='square')

    # # create adversarial example
    # adversary = AutoAttack(net, norm='Linf', eps=8/255, version='standard')
    # # adversary.attacks_to_run = ['square']
    # for batch_idx, (inputs, targets) in enumerate(testloader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         dict_adv = adversary.run_standard_evaluation_individual(inputs, targets, bs=bs)
    #         print(type(dict_adv))
    #         print(inputs.shape)
    #         torch.set_printoptions(edgeitems=5000)
    #         print(dict_adv['apgd-ce'][8])
    #         print("#############################################")
    #         print("#############################################")
    #         print("#############################################")
    #         print("#############################################")
    #         print("#############################################")
    #         print(dict_adv['apgd-t'][8])
    #         print("#############################################")
    #         print("#############################################")
    #         print("#############################################")
    #         print("#############################################")
    #         print("#############################################")
    #         print(dict_adv['fab-t'][8])
    #         print("#############################################")
    #         print("#############################################")
    #         print("#############################################")
    #         print("#############################################")
    #         print("#############################################")
    #         print(dict_adv['square'][8])
    #         # print(dict_adv['apgd-t'].shape)
    #         # print(dict_adv['fab-t'].shape)
    #         # print(dict_adv['square'].shape)
    #         break

    # for i in range(101):
    #     print(i)
    #     transform_test = transforms.Compose([
    #         transforms.Resize(size),
    #         transforms.ToTensor(),
    #         # block_scrambling(),
    #         pixel_shuffling_new(seed=i),
    #         bit_flipping_new(seed=i),
    #         # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         # pixel_shuffling(),
    #     ])
    #     testset = torchvision.datasets.CIFAR10(root='../../../data', train=False, download=False, transform=transform_test)
    #     testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)

    #     val_loss, acc = test(epoch)
    #     list_acc.append(acc)
    # print(len(list_acc))
    # print(list_acc)
    break
    
    # if args.cos:
    #     scheduler.step(epoch-1)
    
    # list_loss.append(val_loss)
    # list_acc.append(acc)
    
    # # Log training..
    # wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
    #     "epoch_time": time.time()-start})

    # # Write out csv..
    # with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n')
    #     writer.writerow(list_loss) 
    #     writer.writerow(list_acc) 
    # print(list_loss)

# # writeout wandb
# wandb.save("wandb_{}.h5".format(args.net))
    
