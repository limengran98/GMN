import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
#m = compute_feature_smoothness(X.cpu(),adj.cpu().numpy())
class GNN(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.apha = nn.Parameter(torch.FloatTensor(1))
        torch.nn.init.xavier_uniform_(self.weight)
        #torch.nn.init.xavier_uniform_(self.apha)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, features, adj):

        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        return self.relu(output)
    
class Mask(nn.Module):
    def __init__(self, in_features, out_features):
        super(Mask, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor([0]))
        torch.nn.init.xavier_uniform_(self.A)
        torch.nn.init.xavier_uniform_(self.weight)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = 0.6
        
        
    def normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    def forward(self, X, adj,gcn = False,le = True):
        #low_X = torch.mm(X, self.A)
        Wh = torch.mm(X, self.weight)
        low_X = torch.mm(adj, Wh)
        #X = torch.mm(adj, low_X)
        if gcn == False:
            output= low_X
            loss_X=0
            loss_A=0
            self.target = 0
        else:
        # distance matrix-->proba_matrix
          sum_row = torch.sum(low_X ** 2, dim = 1)
          xxt = torch.mm(low_X, low_X.T)
          pij_mat = sum_row + torch.reshape(sum_row, (-1, 1)) - 2 * xxt     
        # (n_samples, n_samples)
          pij_mat = torch.exp(0.0 - pij_mat)

        #torch.diagonal(pij_mat, 0)
          pij_mat = pij_mat / torch.sum(pij_mat, dim = 1)[:, None]                          # (n_samples, n_samples)
        #diag = torch.diag(pij_mat)
        #a_diag = torch.diag_embed(diag)
        #pij_mat = pij_mat - a_diag
        # mask where mask_{ij} = True if Y[i] == Y[j], shape = (n_samples, n_samples)
        
          mask = adj == 0  # (n_samples, n_samples)
          pij_mat_mask = pij_mat * mask   
          #pij_mat_mask = pij_mat #* mask 
          
          
        # pi = \sum_{j \in C_i} p_{ij}
          pi_arr = torch.sum(pij_mat_mask, dim = 1)                             # (n_samples, )
 
          self.target = self.a * torch.sum(pi_arr)

          #pij_mat_mask = F.softmax(pij_mat_mask,dim=1)
          pij_mat_mask = self.relu(pij_mat_mask)
          #pij_mat_mask = self.normalize(pij_mat_mask)
        
          X1 = torch.mm(pij_mat_mask, Wh)
        #X = torch.mm(adj, Wh)
        #print(self.a)
          #loss_X = F.mse_loss(low_X, X1)
          #loss_A = F.mse_loss(adj, pij_mat_mask)
          #print(loss_X)
          #print(loss_A)
          if True:
            #output = torch.mm((adj+pij_mat_mask), Wh)
            output=  X1*self.a +low_X
          else:
            output=  low_X#*self.a +low_X
        #loss_A = F.kl_div(adj, pij_mat_mask, reduction='batchmean')  
        return self.target, output 
        

    
    
class GMN(nn.Module):

    def __init__(self, in_features,hide_features ,out_features):
        super(GMN, self).__init__()

        self.hgac1 = GNN(in_features,hide_features)
        self.hgac2 = GNN(hide_features ,out_features)
        self.nca1 = Mask(in_features,hide_features)
        self.nca2 = Mask(hide_features ,out_features)
        self.mlp = nn.Linear(hide_features ,out_features)

                
    def forward(self, X, H):
        nca_loss1=0
        nca_loss2=0
        nca_loss1,X = self.nca1(X, H)
        #X = self.hgac1(X,H)
        #z = self.hgac2(X,H)
        nca_loss2,z = self.nca2(X, H)
        #X = F.dropout(X, 0.8, training=self.training)
        #z = self.mlp(X)   #1

        return z,nca_loss1+nca_loss2
