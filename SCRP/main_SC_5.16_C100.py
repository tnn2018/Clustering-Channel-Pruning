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
from sklearn.cluster import KMeans as KM
import time 
import gc
from memory_profiler import profile

#import nvgraph
#import libnvgraph
# usr/local/cuda/include/nvgraph.h
# usr/local/cuda/lib64/libnvgraph.so
#from nvGRAPH import SpectralClusteringParameter, nvgraphSpectralClustering, nvgraphAnalyzeClustering
#from cluster123.prcs.parallel.clustering_gpu import clustering_gpu
#import clustering_gpu 
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', # ./logs300/checkpoint.pth.tar
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs300/20200603_C100/', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#assert 1==2
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
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
#print(model)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 9, 'M1']
#cfg = [64, 64, 'M', 128, 128, 'M', 256, 36, 36, 36, 'M', 36, 36, 36, 36, 'M', 9, 9, 9, 9, 'M1'] # 96.43%
#cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 128, 128, 'M', 72, 72, 72, 72, 'M', 36, 36, 36, 36, 'M1'] # 90.69%
cfg = [64, 64, 'M', 128, 128, 'M', 256, 36, 36, 'M', 36, 36, 36, 'M', 9, 9, 9]
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

def Normalize(data):
    m = torch.mean(data).float()
    mx = torch.max(data).float()
    mn = torch.min(data).float()
    return (torch.cuda.FloatTensor(data) - m) / (mx - mn)

def Normalizenew(data):
    m = torch.sum(data.pow(2))
    m = m.sqrt()
    return (torch.cuda.FloatTensor(data)/m)
  #  return data
def Normalizecpu(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return [(float(i) - m) / (mx - mn) for i in data ]

#@profile
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
        
        ##########******************* Spectral Clustering Regularization Loss **********2020-1-6 **********###########
        gamma_1 = torch.tensor(1.0).cuda().float()
        gamma_2 = torch.tensor(2.0).cuda().float()
        theta_1 = torch.tensor(1e-4).cuda().float()
      #  theta_2 = torch.tensor(1e-4).cuda().float()

        count = 0  
      #  SC_loss = 0
        for m in model.modules():
       #   loss = 0
          if isinstance(m, nn.Conv2d):
              out_channels = m.weight.data.shape[0]
              if out_channels == cfg[count]:
                count += 1 
                continue
              
              weight_copy = m.weight.data.clone()
              weight_copy = weight_copy.view([weight_copy.shape[0],-1])

              ########## GPU ###########

        #      for i in  range(out_channels):
        #    
        #         weight_copy[i,:] = Normalizenew(weight_copy[i,:])
              #   m = torch.sum(weight_copy[i,:].pow(2))
              #   m = m.sqrt()
              #   weight_copy[i,:] =torch.cuda.FloatTensor( weight_copy[i,:] )/m

              ########## CPU ############
        #      weight = weight_copy[:, 0:m.weight.data.shape[1]]
        #      weight = weight.cpu().numpy() 
              
              S = torch.zeros((out_channels,out_channels)).cuda()
            #  D = torch.zeros((out_channels,out_channels)).cuda()
           #   for i in range(out_channels):
           #        for j in range(out_channels):
           #          S[i,j] = torch.exp(-gamma_1 * torch.sum( torch.pow((weight_copy[i]-weight_copy[j]), 2) ) ) 
            #         S[i,j] = torch.exp(-gamma_1 * torch.sum( torch.pow((weight_copy[i,:]-weight_copy[j,:]), 2) ) ) 
              #############fast Calculation with ##############
              T = torch.cuda.FloatTensor(np.ones((out_channels,out_channels)))   # all 1
              S = torch.exp(-gamma_2*( T - torch.mm(weight_copy,weight_copy.t()) )) 
              ############## fast Calculation without ##################################
              # T = torch.cuda.FloatTensor(out_channels,1) 
              # for i in  range(out_channels):
              #     T[i] = torch.sum( torch.pow(weight_copy[i,:],2) )
              # T_1 = T.repeat(1, out_channels)  
              # T_2 = T_1.t()
              # T_3 = T_1 + T_2
              # S = torch.exp(-T_3 + gamma_2*(torch.mm(weight_copy,weight_copy.t()) )) 
              #######################################################################
            #  D = torch.diag(torch.sum(S,1) )
            #  L = D - S 
             
            #  clustering = SC(n_clusters=cfg[count],eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf',
            #  n_neighbors=10, eigen_tol=0.0, assign_labels='discretize',degree=3, coef0=1,kernel_params=None, n_jobs=None).fit(weight)
              #print('clustering.labels_: \n {}\n'.format(clustering.labels_)) 'discretize'  'kmeans'

              ############################################
            #  H = np.zeros((out_channels,cfg[count]))
            #  for i in range(out_channels):
            #      j = clustering.labels_[i]
            #      H[i,j] = 1
            #  I1 = np.eye(cfg[count])
              I2 = np.eye(out_channels) # new

            #  I1 = torch.cuda.FloatTensor(I1)
              I2 = torch.cuda.FloatTensor(I2)
            #  H = torch.cuda.FloatTensor(H) 

            #  loss = torch.mm(torch.mm(H.t(),L),H)
            #  loss = torch.trae(loss)
        
              ############## version acc cuda########################
              S = torch.cuda.FloatTensor(S)
              S = torch.mul(S, T.sub(I2)) # S*
              S1 = torch.t(torch.cuda.FloatTensor(weight_copy))
              grad = torch.mm(S1,S) # WS
            #  grad_2 = torch.sum(S,dim=1)
            #  grad_3 = S1
             
            #  for i in range(out_channels):

            #      grad_3[:,i] = grad_2[i]*S1[:,i]
              
            #  grad_4 = grad - grad_3
        
              grad = torch.t(grad).reshape(out_channels,m.weight.data.shape[1],3,3)  # grad_4 Accurate \\ grad Approximate
            #  grad = torch.t(grad).reshape(out_channels,m.weight.data.shape[1],3,3)
              ######################################
              m.weight.grad.data += theta_1*grad
            
              count += 1
          elif isinstance(m, nn.MaxPool2d):
           #   loss = 0
              count += 1
          else :
              continue    
        #  SC_loss += loss         
      #  for x in locals().keys():
          #   print(locals()[x])
          #   del locals()[x]
       #   del weight_copy
       #   gc.collect()   
        
##########******************* Spectral Clustering Regularization Loss *************END*************###########
      #  loss = loss + (theta_1/4)*SC_loss
      ########## add SC_loss #####  
        avg_loss += loss
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        #loss.backward()
                               
        ###############################################################        
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print(time.time()-start_time)
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))
                #100. * batch_idx / len(train_loader), loss.item()))
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
           #test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss 
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
#print(test())
best_prec1 = 0.
is_best = False
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec1 = test()
    print(prec1)
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
