import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import *

from sklearn.cluster import SpectralClustering as SC
##########################   Spectral Clustering  Calculating intra-class average  ###############
# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=32,
                    help='depth of the resnet')
parser.add_argument('--model', default='./ResNet/resnet32-c10/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./ResNet/resnet32-c10/4/', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                     help='random seed (default: 1)')                                          
parser.add_argument('-v', default='B', type=str, 
                    help='version of the model')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)  
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = resnet(depth=args.depth, dataset=args.dataset)
#print(model)
if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print('Pre-processing Successful!')
#print(model)
# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
               data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)   
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        return correct / float(len(test_loader.dataset))

acc = test(model)

skip = {
    #'A': [10, 12, 20, 22, 30],  # 1-40epoch
    #'A': [10],   # 3-60epoch
    'B': [],  # 2-400-epoch  0.9 0.9 0.9
}

prune_prob = {
    #'A': [0.1, 0.3, 0.3],  
    #'A': [0.5, 0.7, 0.9],
    'B': [0.7, 0.8, 0.9],  # 4-40epoch
}
cfg = []
layer_id = 1
cfg_mask = {}
for m in model.modules():
    if isinstance(m, nn.Conv2d): 
        out_channels = m.weight.data.shape[0]
        if layer_id in skip[args.v]:   
            cfg_mask[layer_id] = torch.ones(out_channels)
            cfg.append(out_channels)
            #print(cfg_mask.keys())
            layer_id += 1
            continue
        if layer_id % 2 == 0:
            if layer_id <= 18:
                stage = 0
            elif layer_id <= 36:
                stage = 1
            else: 
                stage = 2
            prune_prob_stage = prune_prob[args.v][stage]
            num_keep = int(out_channels * (1 - prune_prob_stage))  
            cfg.append(num_keep)

            weight_norm = m.weight.data.abs().clone().cpu().numpy()
            L1_norm = np.sum(weight_norm, axis=(1,2,3))  
            #arg_max = np.argsort(L1_norm)
            #arg_max_rev = arg_max[::-1][:num_keep]
            #mask = torch.zeros(out_channels)
            #mask[arg_max_rev.tolist()] = 1
            #cfg_mask.append(mask)
            #cfg.append(num_keep)
            weight_copy = m.weight.data.clone()
            weight_copy = weight_copy.view([weight_copy.shape[0],-1])
            weight_copy = weight_copy.cpu().numpy() 
            clustering = SC(n_clusters=num_keep,eigen_solver=None, random_state=None, n_init=10, gamma=1.0, 
              affinity='rbf',n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans',degree=3, coef0=1,kernel_params=None, n_jobs=None).fit(weight_copy)
            arg_max = np.argsort(clustering.labels_.tolist())  
            mask2 = {}    
            co = 0
            mask3 = []
            for id in arg_max: 
                now_label = clustering.labels_.tolist()[int(id)]
                if now_label == co:             
                   mask2[co] = mask3
                   mask3.append(id)
                   continue  
                mask3=[id]
                mask2[co+1] = mask3 
                co += 1       
            keys = mask2.keys()
            assert len(keys) == num_keep, "size of arg_max_rev not correct"                        
            cfg_mask[layer_id] = mask2  #dict           
            layer_id += 1
            continue
        layer_id += 1

# *************************************************
print('*********************************8')
keys = cfg_mask.keys()
print('length of keys: {}'.format(len(keys)))
print('cfg_mask keys:{}'.format(keys))
#for key in keys:
#    value = cfg_mask[key]
#    print(key)
#    print(value)
print('cfg:{}'.format(cfg))
print('len of cfg:{}'.format(len(cfg)))
print('*********************************8') 

newmodel = resnet(dataset=args.dataset, depth=args.depth,cfg=cfg)
if args.cuda:
    newmodel.cuda()
#print(newmodel)
layer_id_in_cfg = 1
conv_count = 1

for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.Conv2d):
        print('nn.Conv2d')
        if conv_count == 1:
            print(conv_count)
            print(layer_id_in_cfg)
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            layer_id_in_cfg += 1
            continue            
        if conv_count % 2 == 0:  
          print(conv_count)  
          print(layer_id_in_cfg)
          if conv_count in skip[args.v]:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            layer_id_in_cfg += 1
            continue 
          end_mask = cfg_mask[layer_id_in_cfg]  
          
          w1 = m0.weight.data.clone()   
          end_keys = end_mask.keys() 
          idx1_end = len(end_keys)
          weight_all_end = [] 
          print(idx1_end)   
          for key in end_keys :
              end_value = end_mask[key] # end_value is a list  
              idx2 = len(end_value)
              count = 0   
              for id in range(idx2) :
                  id_value = end_value[id]
                  count += 1 
                  if  count == 1:                    
                      numpy = w1[id_value, :, :, :].clone().cpu().numpy()
                  else :
                      numpy += w1[id_value, :, :, :].clone().cpu().numpy()
              numpy = numpy / idx2        
              weight_all_end.append(numpy)
          print(len(weight_all_end))               
          m1.weight.data = torch.Tensor(weight_all_end).cuda()
          print(m1.weight.data.size())
          layer_id_in_cfg += 1
          conv_count += 1
          continue
        
        if conv_count % 2 == 1:
          print(conv_count)
          print(layer_id_in_cfg) 
          if conv_count-1 in skip[args.v]:
            m1.weight.data = m0.weight.data.clone()  
            conv_count += 1
            layer_id_in_cfg += 1
            continue
          mask = cfg_mask[layer_id_in_cfg-1]
    
          w1 = m0.weight.data.clone()   
          keys = mask.keys() 
          idx1 = len(keys)
          print(idx1)
          weight_all = []    
          for key in keys :
              value = mask[key] # end_value is a list  
              idx2 = len(value)
              count = 0   
              for id in range(idx2) :
                  id_value = value[id]
                  count += 1 
                  if  count == 1:                    
                      numpy = w1[:, id_value, :, :].clone().cpu().numpy()
                  else :
                      numpy += w1[:, id_value, :, :].clone().cpu().numpy()
              numpy = numpy / idx2        
              weight_all.append(numpy)
          print(len(weight_all))               
          m1.weight.data = torch.Tensor(weight_all).permute(1,0,2,3).cuda()    
          print(m1.weight.data.size())
          layer_id_in_cfg += 1           
          conv_count += 1
          continue
    elif isinstance(m0, nn.BatchNorm2d):
        print('nn.BatchNorm2d')
        if conv_count % 2 == 1:
          print(conv_count-1)
          print(layer_id_in_cfg-1) 
          if conv_count-1 in skip[args.v]:
             m1.weight.data = m0.weight.data.clone()
             m1.bias.data = m0.bias.data.clone()
             m1.running_mean = m0.running_mean.clone()
             m1.running_var = m0.running_var.clone()  
             continue  
          mask = cfg_mask[layer_id_in_cfg-1]  
          #print(conv_count-1)
          #print(layer_id_in_cfg-1)   
          keys = mask.keys() 
          idx1 = len(keys)
          weight_all = []  
          bias_all = []
          running_mean_all = []
          running_var_all = []
          for key in keys :
              value = mask[key] # end_value is a list  
              idx2 = len(value)
              count = 0   
              for id in range(idx2) :
                  id_value = value[id]
                  count += 1 
                  if  count == 1:                    
                      weight = m0.weight.data[id_value].clone()
                      bias =  m0.bias.data[id_value].clone()
                      running_mean = m0.running_mean[id_value].clone()
                      running_var =  m0.running_var[id_value].clone()
                  else :
                      weight +=  m0.weight.data[id_value].clone()
                      bias += m0.bias.data[id_value].clone()
                      running_mean +=  m0.running_mean[id_value].clone()
                      running_var +=  m0.running_var[id_value].clone()
                    
              weight = weight / idx2        
              weight_all.append(weight)  
              bias = bias / idx2 
              bias_all.append(bias)   
              running_mean = running_mean / idx2   
              running_mean_all.append(running_mean)
              running_var = running_var / idx2    
              running_var_all.append(running_var)           
          m1.weight.data = torch.Tensor(weight_all).cuda() 
          m1.bias.data = torch.Tensor(bias_all).cuda()
          m1.running_mean = torch.Tensor(running_mean_all).cuda() 
          m1.running_var = torch.Tensor(running_var_all).cuda()            
          continue  
        print(conv_count-1)  
        print(layer_id_in_cfg-1)   
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Linear):
        print('nn.Linear')
        print(conv_count)
        print(layer_id_in_cfg)
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()


##########################################################

#torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
torch.save({'cfg': cfg,'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
num_parameters = sum([param.nelement() for param in newmodel.parameters()])
num_parameters_u = sum([param.nelement() for param in model.parameters()])
print(newmodel)
model = newmodel
acc = test(model)

print("number of parameters: "+str(num_parameters))
print("number of num_parameters_u: "+str(num_parameters_u))
with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc)+"\n")
