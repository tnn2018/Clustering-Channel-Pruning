from __future__ import print_function
import argparse
import numpy as np
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
from sklearn.cluster import SpectralClustering as SC
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',   # (epoch 160)
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default=' ', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./ResNet/resnet32-c10/SC29_2/', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='resnet', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the neural network')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-v', default='B', type=str, 
                    help='version of the model')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
print(model)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
print(model)
skip = {
    #'A': [10, 12, 20, 22, 30],  
    #'A': [10],   
    'B': [],  
}

prune_prob = {
    #'A': [0.1, 0.3, 0.3],  
    #'A': [0.5, 0.7, 0.9],
  #  'B': [0.7, 0.8, 0.9],  # 87.72%  acc=87.69%
    'B': [0.2, 0.2, 0.6],  # 
}
# cfg = []
# layer_id = 1
def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        ##########******************* Spectral Clustering Regularization Loss **********2020-1-16 **********###########
        gamma_1 = torch.tensor(1.0).cuda().float()
        gamma_2 = torch.tensor(2.0).cuda().float()
        theta_1 = torch.tensor(1e-4).cuda().float()

        cfg = []
        layer_id = 1
        for m in model.modules():
          if isinstance(m, nn.Conv2d):
              out_channels = m.weight.data.shape[0]
              if layer_id in skip[args.v]: 
                 cfg.append(out_channels),                 
                 layer_id += 1
                 continue
              if layer_id % 2 == 0:
                if layer_id <= 10:   # 18
                   stage = 0
                elif layer_id <= 20:  #  36
                   stage = 1
                else: 
                   stage = 2
                prune_prob_stage = prune_prob[args.v][stage]
                num_keep = int(out_channels * (1 - prune_prob_stage))  
                cfg.append(num_keep)
                
                weight_copy = m.weight.data.clone()
                weight_copy = weight_copy.view([weight_copy.shape[0],-1])
                
                S = torch.zeros((out_channels,out_channels)).cuda()  # GPU S 
                T = torch.cuda.FloatTensor(np.ones((out_channels,out_channels)))   # all 1
                S = torch.exp(-gamma_2*( T - torch.mm(weight_copy,weight_copy.t()) ))  # Calculate similarity matrix
                
                I = np.eye(out_channels)   
                I = torch.cuda.FloatTensor(I)
                
                S = torch.cuda.FloatTensor(S)
                S = torch.mul(S, T.sub(I)) # S*
                S1 = torch.t(torch.cuda.FloatTensor(weight_copy))
                grad = torch.mm(S1,S) 
                grad = torch.t(grad).reshape(out_channels,m.weight.data.shape[1],3,3)   # compute gradient

                m.weight.grad.data += theta_1*grad
                count += 1
  ##############***************#############################################################*********#############
        avg_loss += loss
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()  

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(time.time()-start_time)
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    re = 0
    with torch.no_grad():
        for data, target in test_loader:
           if args.cuda:
              data, target = data.cuda(), target.cuda()
           data, target = Variable(data), Variable(target)
           output = model(data)
#           test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss 
           test_loss += F.cross_entropy(output, target, reduction='sum').item()  # new 
           pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
           correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        re =  correct
        print(re)
        return re

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

best_prec1 = 0.
is_best = False
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec1 = test()
    #print(prec1)
    if prec1 > best_prec1:
       is_best = True
    #print(is_best)
    best_prec1 = max(prec1, best_prec1)
    print(best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        'cfg': model.cfg
    }, is_best, filepath=args.save)
    is_best = False
