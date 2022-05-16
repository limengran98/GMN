import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import InMemoryDataset, download_url

import scipy.sparse as sp
import argparse

from dataset.CoraML import CoraML 
from dataset.DBLP import DBLP 
from dataset.Coauthor import Coauthor 
from dataset.Amazon import Amazon 

from utils.cora_ml import read_cora_ml_data
from utils.other import *
from model import GMN

from sklearn import preprocessing
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,f1_score
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#from pytorch_metric_learning import losses
import warnings
warnings.filterwarnings("ignore")




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,default='Citeseer',
                    help='Dataset')
parser.add_argument('--epoch', type=int, default=200,
                    help='Training Epochs')
parser.add_argument('--node_dim', type=int, default=16,
                    help='Node dimension')
parser.add_argument('--head', type=int, default=1)
parser.add_argument('--dropout', type=int, default=0.8)
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01,
                    help='l2 reg')
parser.add_argument('--metrics', default=True)
parser.add_argument('--mse', default=False)
parser.add_argument('--le', default=True)
parser.add_argument('--k', default=2)
parser.add_argument('--knn', default=False)

args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataname = args.dataset
metrics = args.metrics 
k = args.k 
knn = args.knn 
mse = args.mse
le = args.le
epoch = args.epoch

if dataname == 'Citeseer':
    dataset = Planetoid(root='.', name='Citeseer')
    n_class = 6
if dataname == 'Cora':
    dataset = Planetoid(root='.', name='Cora')
    n_class = 7
if dataname == 'Pubmed':
    dataset = Planetoid(root='.', name='Pubmed')
    n_class = 3
if dataname == 'cora_ml':
    dataset = CoraML(root='./CoraML', name='cora_ml')
    n_class = 7
#dataset = DBLP(root='./dblp', name='dblp')
if dataname == 'photo':
    dataset = Amazon(root='./Amazon', name='photo')
    n_class = 8
if dataname == 'cs':
    dataset = Coauthor (root='./CS', name='cs')
    n_class = 15

data = dataset[0].to(device)
X = data['x']
y = data['y']

train_mask = data['train_mask']
val_mask = data['val_mask']
test_mask = data['test_mask']

if knn:
    A = kneighbors_graph(X.cpu().numpy(), k, mode='connectivity', include_self=True)
    A = A.toarray()
    adj = torch.Tensor(normalize(A)).to(device)
else:
  graph = data.edge_index.T
  adj,a = load_graph(X,graph.cpu())
#adj = generate_HG(X,graph.cpu())
  adj = torch.Tensor(adj).to(device)
  a = torch.Tensor(a).to(device)


model = GMN(in_features=X.shape[1], 
              hide_features=16,
              out_features=n_class).to(device)
            

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)   

# weights = [1,1,1,1,1,1]
# class_weights = torch.FloatTensor(weights)
eprm_state = '_mask'
file_out = open('./output/'+dataname+'_'+eprm_state+'.txt', 'a')
print("The experimental results", file=file_out)
Loss = nn.CrossEntropyLoss()



# loss_func = losses.TripletMarginLoss()
# loss_func = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
# loss_func = losses.ProxyNCALoss(n_class, embedding_size =n_class, softmax_scale=1)
# loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
# losses.CircleLoss(m=0.4, gamma=80)

# loss_func = losses.NPairsLoss()
#loss_func = losses.NCALoss(softmax_scale=1)




best_f1 = 0 
best_acc = 0
best_pre = 0
best_rec = 0
metric = []
lossce = []
lossmse = []
losssum=[]
acclist = []
poslist = []
neglist = []
##
#m = compute_feature_smoothness(X.cpu().numpy(),adj.cpu().numpy())
#print(m)
#metric.append(m)

for i in range(epoch):
    model.zero_grad()
    model.train()
    output,l2 = model(X,adj)
    if True:
        p_mask = a[train_mask][:,train_mask]
        lossc = Loss(output[train_mask],y[train_mask].long())
        #lossm = loss_func(output[train_mask],y[train_mask].long())#
        #lossm = 0
        MSloss ,pos ,neg  = MultiSimilarityLoss(output[train_mask],y[train_mask].long(),p_mask)

        loss =   lossc + MSloss ##+  0.0001*l2
        #print(lossm)
        #print(lossc)
      
    else:
        loss = Loss(output[train_mask],y[train_mask].long())
        #print(torch.exp(-loss_mse))
    #lossce.append(lossc.cpu().detach().numpy())
    #lossmse.append(lossm.cpu().detach().numpy())
    #losssum.append(loss.cpu().detach().numpy())
    #print(l2)
    
    
    #if i%10 ==0:
        #print(accuracy_score(y_train,torch.argmax(output,dim=1).numpy()))
        #print(classification_report(y_train-1,torch.argmax(output,dim=1).numpy(), digits=4))
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        output_test,l2 = model(X,adj)
        f1 = f1_score(y[test_mask].cpu().numpy(),torch.argmax(output_test[test_mask],dim=1).cpu().numpy(),average = 'macro')
        acc = accuracy_score(y[test_mask].cpu().numpy(),torch.argmax(output_test[test_mask],dim=1).cpu().numpy())
        pre = precision_score(y[test_mask].cpu().numpy(),torch.argmax(output_test[test_mask],dim=1).cpu().numpy(),average = 'macro')
        rec = recall_score(y[test_mask].cpu().numpy(),torch.argmax(output_test[test_mask],dim=1).cpu().numpy(),average = 'macro')
        if i%10 ==0:
            print(classification_report(y[test_mask].cpu().numpy(),torch.argmax(output_test[test_mask],dim=1).cpu().numpy(), digits=4))
            #print(classification_report(y[test_mask].cpu().numpy(),torch.argmax(output_test[test_mask],dim=1).cpu().numpy(), digits=4), file=file_out)
            #print(accuracy_score(y_test,torch.argmax(output_test,dim=1).numpy()))
        acclist.append(acc)
        poslist.append(pos.cpu().detach().numpy())
        neglist.append(neg.cpu().detach().numpy())
            #m = compute_feature_smoothness(output_test.cpu(),adj.cpu().numpy())
            #metric.append(m)
        if f1 > best_f1:
            best_f1 = f1 
            best_acc = acc
            best_pre = pre
            best_rec = rec


print(args,file=file_out)
print('Test k: {}, \n  Macro_Pre: {}, \n Macro_Rec: {}, \n Macro_F1: {}, \n Acc: {}'.format(k,  best_pre, best_rec,best_f1, best_acc),file=file_out)
print('Test k: {}, \n  Macro_Pre: {}, \n Macro_Rec: {}, \n Macro_F1: {}, \n Acc: {}'.format(k,  best_pre, best_rec,best_f1, best_acc))
