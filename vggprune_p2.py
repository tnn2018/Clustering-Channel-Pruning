import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import *

from sklearn.cluster import SpectralClustering as SC

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--model', default='./logs300/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./vgg300/1/', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                     help='random seed (default: 1)')                                          

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # no effect

torch.manual_seed(args.seed)  
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = vgg(dataset=args.dataset, depth=args.depth)
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

num_before = sum([param.nelement() for param in model.parameters()])
print(num_before)
# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False,download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
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
# cfg = [32, 64, 'M', 64, 128, 'M', 128, 256, 256, 'M', 256, 512, 512, 'M', 512, 512, 512] # 46%  vgg-16
# cfg = [64, 64, 'M', 128, 128, 'M', 128, 256, 256, 'M', 300, 400, 400, 'M', 400, 512, 512] vgg-16 60% 
# cfg = [32, 64, 'M', 64, 128, 'M', 128, 128, 256, 'M', 256, 256, 512, 'M', 512, 512, 512] # 30%  vgg-16
# cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512] # 93% vgg-13
# cfg = [32, 64, 'M', 64, 128, 'M', 128, 256, 'M', 256, 512, 'M', 512, 512] # 66%  vgg-13
# cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512] # vgg19
# cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512] # vgg11

# cfg = [64, 64, 'M', 128, 128, 'M', 200, 256, 'M', 300, 400, 'M', 512, 512] # vgg-13 80% Average 
# cfg = [64, 64, 'M', 128, 128, 'M', 200, 256, 'M', 300, 300, 'M', 512, 512]

#cfg = [64, 64, 'M', 128, 128, 'M', 205, 256, 256, 'M', 307 , 358 , 358, 'M', 410, 461, 461]
#cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M1']  # new vgg-16
#cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 128, 'M', 36, 36, 36, 36, 'M', 9, 9, 9, 9, 'M1' ]  # 2.1M 93.90% 
#cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 128, 128, 128, 128, 'M', 9, 9, 9, 9, 'M1' ] # 3.1M 
cfg = [64, 64, 'M', 128, 128, 'M', 256, 72, 72, 72, 'M', 36, 36, 36, 36, 'M', 9, 9, 9, 9, 'M1'] # 

##### choose the Average in each class ##########
cfg_mask = {}
layer_id = 0 
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        print(m)
        out_channels = m.weight.data.shape[0]
        if out_channels == cfg[layer_id]:
            cfg_mask[layer_id] = torch.ones(out_channels) #torch.Tensor
            #print('cfg_mask[{}]:{}'.format(layer_id,cfg_mask[layer_id])) 
            layer_id += 1

            continue 
        #print('*************************')
        weight_copy = m.weight.data.clone()
        #print(weight_copy.shape) 
        weight_norm = m.weight.data.abs().clone()
        weight_norm = weight_norm.cpu().numpy()
        L1_norm = np.sum(weight_norm, axis=(1, 2, 3))
        #print(weight_copy.shape)   
        weight_copy = weight_copy.view([weight_copy.shape[0],-1])
        weight_copy = weight_copy.cpu().numpy() 
        #print(weight_copy.shape)
        clustering = SC(n_clusters=cfg[layer_id],eigen_solver=None, random_state=None, n_init=10, gamma=1.0, 
              affinity='rbf',n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans',degree=3, coef0=1,kernel_params=None, n_jobs=None).fit(weight_copy)
        #print('clustering.labels_: \n {}\n'.format(clustering.labels_))  
        arg_max = np.argsort(clustering.labels_.tolist())  
        #print('arg_max : \n {} \n'.format(arg_max))
        mask2 = {}    
        co = 0
        mask3 = []
        for id in arg_max: 
           now_label = clustering.labels_.tolist()[int(id)]
           #print('now_label:{}\t {}\n'.format(id,now_label))

           if now_label == co:
              
              mask2[co] = mask3
              mask3.append(id)
              continue  
           mask3=[id]
           mask2[co+1] = mask3 
           #print('*************mask2:\n {}\n mask3:\n{}\n'.format(mask2,mask3))
           co += 1
        
        keys = mask2.keys()
        #print(len(keys))
        assert len(keys) == cfg[layer_id], "size of arg_max_rev not correct" 
        #print('mask2:\n{}\n'.format(mask2)) 
                       
        cfg_mask[layer_id] = mask2  #dict
        layer_id += 1      
    elif isinstance(m, nn.MaxPool2d):
        layer_id += 1
    elif isinstance(m, nn.AvgPool2d):
        #layer_id += 1
        continue
#print('*********************************8')
#cfg_keys = cfg_mask.keys()
# print(len(cfg_keys))
# print(cfg_keys)
#for keys in cfg_keys:
#    value = cfg_keys[keys]
#    print(keys)
#    print(value)
#print('*********************************8') 
############################################################

newmodel = vgg(dataset=args.dataset, cfg=cfg)
if args.cuda:
    newmodel.cuda()

start_mask = torch.ones(3) # torch.Tensor
layer_id_in_cfg = 0
end_mask = cfg_mask[layer_id_in_cfg]
   
for [m0, m1] in zip(model.modules(), newmodel.modules()):
  #print('******layer_id_in_cfg: \t {}********'.format(layer_id_in_cfg)) 
  #print(m0)
  if isinstance(end_mask,torch.Tensor):
    #print('end_mask: \t torch.Tensor') # print class
    if isinstance(m0, nn.BatchNorm2d):
        #print('nn.BatchNorm2d')   
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) # end_mask
        if idx1.size == 1: 
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        #print('Out shape {:d}'.format(idx1.size))  
        layer_id_in_cfg += 1 
        #print(end_mask) # end_mask not update
        start_mask = end_mask  # not update
        if layer_id_in_cfg < len(cfg):  # do not change in Final FC
            #if layer_id_in_cfg in [2,5,8,11]:  # vgg13 only 
            #if layer_id_in_cfg in [2,5,9,13]:  # vgg16 only
            if layer_id_in_cfg in [2,5,10,15]:  # vgg_new only
               layer_id_in_cfg += 1 
            elif layer_id_in_cfg in [20]:  # vgg_new only
               layer_id_in_cfg -= 1    
            end_mask = cfg_mask[layer_id_in_cfg] # update  layer_id_in_cfg=13 start=12 end=13    
        #print(end_mask)               
    elif isinstance(m0, nn.Conv2d):
      #print('nn.Conv2d') 
      if isinstance(start_mask,torch.Tensor):
        #print('start_mask: torch.Tensor')
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy()))) # start_mask
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))   # end_mask
        #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
      elif isinstance(start_mask,dict):       
          #print('start_mask: dict')
          start_keys = start_mask.keys()
          idx1 = len(start_keys)
          weight_all = []             
          for key in start_keys :
              start_value = start_mask[key] # list
              idx2 = len(start_value)
              count = 0
              for id in range(idx2) :
                id_value = start_value[id]
                count += 1
                #print(count)
                if count == 1 :
                   #print(m0.weight.data.size())
                   weight = m0.weight.data[:, id_value, :, : ].clone().cpu().numpy() #tensor to numpy
                else :
                   #print(m0.weight.data.size())
                   weight += m0.weight.data[:, id_value, :, : ].clone().cpu().numpy()
              weight = weight / idx2
              weight_all.append(weight)
              #print(len(weight_all))
          w1 = torch.Tensor(weight_all)  # list to tensor              
          #print(w1.size()) 
          w1 = w1.permute(1,0,2,3)
          #print(w1.size())
          idx1_end = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
          if idx1_end.size == 1:
             idx1_end = np.resize(idx1_end, (1,))
          w1 = w1[idx1_end.tolist(), :, :, :].clone()
          m1.weight.data = w1.clone().cuda() 
          
          #print('In shape: {:d}, Out shape {:d}.'.format(idx1, idx1_end.size))
      
    elif isinstance(m0, nn.Linear):
        #print('nn.Linear') 
        if layer_id_in_cfg == len(cfg):
            idx0 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) # end_mask the last one
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
            layer_id_in_cfg += 1
            continue
        m1.weight.data = m0.weight.data.clone() # layer_id_in_cfg=14  start=12 end=13 copy
        m1.bias.data = m0.bias.data.clone() 
    elif isinstance(m0, nn.BatchNorm1d):
        #print('BatchNorm1d')
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()

  elif isinstance(end_mask, dict):  #  dict 
     #print('end_mask:\t dict ')  
     if isinstance(m0, nn.BatchNorm2d):  # end_mask
        #print('BatchNorm2d')
        keys = end_mask.keys() 
        idx1 = len(keys) 
        weight_all = []
        bias_all = []
        running_mean_all = []
        running_var_all = []
        for key in keys :          
          value = end_mask[key]  # value is a list 
          idx2 = len(value)      
          #print(value)  
          count = 0 
          for id in range(idx2) :
             id_value = value[id] 
             count += 1 
             if count == 1 : 
               weight = m0.weight.data[id_value].clone()  # tensor to numpy
               bias =  m0.bias.data[id_value].clone()
               running_mean = m0.running_mean[id_value].clone()
               running_var =  m0.running_var[id_value].clone()
               #print('weight: {}\n bias:{}\n running_mean:{}\n running_var:{}\n'.format(weight,bias,running_mean,running_var))
             else :
               #print(m0.weight.data[id_value].clone())
               #print(m0.bias.data[id_value].clone())
               #print(m0.running_mean[id_value].clone())
               #print(m0.running_var[id_value].clone())
               weight +=  m0.weight.data[id_value].clone()
               bias += m0.bias.data[id_value].clone()
               running_mean +=  m0.running_mean[id_value].clone()
               running_var +=  m0.running_var[id_value].clone()
               #print('weight: {}\n bias:{}\n running_mean:{}\n running_var:{}\n'.format(weight,bias,running_mean,running_var))
          #print(weight)
          #print(idx2)
          weight = weight / idx2
          #print(weight)
          weight_all.append(weight)
          #print(len(weight_all))
          bias = bias / idx2
          bias_all.append(bias) 
          running_mean = running_mean / idx2 
          running_mean_all.append(running_mean)
          running_var = running_var / idx2
          running_var_all.append(running_var)
          #print('weight_all: {}'.format(weight_all))
        m1.weight.data = torch.Tensor(weight_all).cuda() # list to tensor
        m1.bias.data = torch.Tensor(bias_all).cuda()
        m1.running_mean = torch.Tensor(running_mean_all).cuda()
        m1.running_var = torch.Tensor(running_var_all).cuda()
        #print(m1.weight.data)
        layer_id_in_cfg += 1
        start_mask = end_mask
        if layer_id_in_cfg < len(cfg):  # do not change in Final FC
           #if layer_id_in_cfg in [2,5,8,11]: # vgg13 only
           #if layer_id_in_cfg in [2,5,9,13]:  # vgg16 only
              #layer_id_in_cfg += 1
           if layer_id_in_cfg in [2,5,10,15]:  # vgg16_new only
              layer_id_in_cfg += 1
              end_mask = cfg_mask[layer_id_in_cfg]
           elif layer_id_in_cfg in [20]:  # vgg16_new only
              layer_id_in_cfg -= 1      
              end_mask = 256
     elif isinstance(m0, nn.Conv2d):  # start_mask  end_mask 
       #print('Conv2d')
       if isinstance(start_mask, dict):   
          #print('start_mask: dict')
          start_keys = start_mask.keys()
          idx1 = len(start_keys)  
          weight_all = [] 
          for key in start_keys : 
              start_value = start_mask[key] # list
              idx2 = len(start_value)  
              count = 0
              for id in range(idx2) : 
                id_value = start_value[id]
                count += 1 
                if count == 1 :
                   weight = m0.weight.data[:, id_value, :, : ].clone().cpu().numpy()  # tensor to numpy
                else :   
                   weight += m0.weight.data[:, id_value, :, : ].clone().cpu().numpy()
              weight = weight / idx2
              weight_all.append(weight)
          w1 = torch.Tensor(weight_all).cuda() # list to tensor to cuda tensor   
          w1 = w1.permute(1,0,2,3)
          end_keys = end_mask.keys()
          idx1_end = len(end_keys)
          weight_all_end = []  
          for key in end_keys :
              end_value = end_mask[key] # list
              idx2 = len(end_value)
              count = 0  
              for id in range(idx2) :
                id_value = end_value[id]
                count += 1                         
                if count == 1 :   
                   weight = w1[id_value, :, :, :].clone().cpu().numpy()
                else :         
                   weight += w1[id_value, :, :, :].clone().cpu().numpy()
              weight = weight / idx2   
              weight_all_end.append(weight)  # list
          m1.weight.data = torch.Tensor(weight_all_end).cuda()
          #print('In shape: {:d}, Out shape {:d}.'.format(idx1, idx1_end))         
        
       elif isinstance(start_mask, torch.Tensor) :
          #print('start_mask: torch.Tensor')  
          idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))   
          if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
          w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()   
          end_keys = end_mask.keys() 
          idx1_end = len(end_keys)
          #assert idx1_end == 32
          weight_all_end = []    
          for key in end_keys :
              end_value = end_mask[key] # end_value is a list  
              #print('end_value:{}'.format(end_value))
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
              #print(len(weight_all_end))
              
          m1.weight.data = torch.Tensor(weight_all_end).cuda()
          #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1_end))
 
     elif isinstance(m0, nn.Linear):   # end_mask
        #print('nn.Linear')
        if layer_id_in_cfg == len(cfg):
            end_mask = cfg_mask[layer_id_in_cfg-1]
            print('&&&&&&&&&&&&&&&&&')
            
            end_keys = end_mask.keys()
            idx1 = len(end_keys)
            weight_all = []
            bias_all = [] 
            for key in end_keys :
               end_value = end_mask[key] # list  
               idx2 = len(end_value) 
               count = 0
               for id in range(idx2) :
                 id_value = end_value[id]
                 count += 1
                 if count == 1 :
                    weight = m0.weight.data[:, id_value].clone().cpu().numpy()
               else :
                    weight += m0.weight.data[:, id_value].clone().cpu().numpy()
               weight = weight / idx2    
               weight_all.append(weight) 
            
            m1.weight.data = torch.Tensor(weight_all).cuda() 
            m1.bias.data = m0.bias.data.clone()
            layer_id_in_cfg += 1
            print(m1.weight.data.size())
            m1.weight.data = m1.weight.data.transpose(0,1)
            print(m1.weight.data.size())
            continue
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        #print(m1.weight.data.size())
        #print(m1.bias.data.size())
     elif isinstance(m0, nn.BatchNorm1d):
        #print('nn.BatchNorm1d')      
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone() 
        #print(m1.weight.data.size())       
        #print(m1.bias.data.size())

#############################################################

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
#print(newmodel)
model = newmodel
acc = test(model)
print(model)
num_parameters = sum([param.nelement() for param in newmodel.parameters()])
print(num_parameters)
with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc)+"\n")
