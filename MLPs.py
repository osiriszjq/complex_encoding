import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# regular MLP
class MLP(nn.Module):
    def __init__(self,input_dim=2, output_dim=3, depth = 0, width= 256,bias=True,use_sigmoid = True):
        super(MLP, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.mm = nn.ModuleList([])
        if depth == 0:
            self.mm.append(nn.Linear(input_dim, output_dim,bias=bias))
        else:
            self.mm.append(nn.Sequential(nn.Linear(input_dim, width,bias=bias),nn.ReLU(True)))
            for i in range(depth-1):
                self.mm.append(nn.Sequential(nn.Linear(width, width,bias=bias),nn.ReLU(True)))
            self.mm.append(nn.Sequential(nn.Linear(width, output_dim,bias=bias)))
        if use_sigmoid: self.mm.append(nn.Sigmoid())
    def forward(self, x):
        for m in self.mm:
            x = m(x)
        return x
    def name(self):
        return "MLP"


# If the input to MLP is in form of kron(x,y), this model can calculate it efficiently.
class Kron_MLP(nn.Module):
    def __init__(self,input_dim=2, output_dim=3, depth=0, width0=256, width=256, bias=True, use_sigmoid=True):
        super(Kron_MLP, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        if depth==0: width0 = output_dim
        
        self.mm = nn.ModuleList([])
        self.first = nn.ParameterDict({
                'weight': nn.Parameter(2/np.sqrt(width0)*torch.rand(width0, input_dim, input_dim)-1/np.sqrt(width0))})
        
        if depth == 1:
            self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, output_dim,bias=bias)))
        if depth > 1:
            self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, width,bias=bias),nn.ReLU(True)))
            for i in range(depth-2):
                self.mm.append(nn.Sequential(nn.Linear(width, width,bias=bias),nn.ReLU(True)))
            self.mm.append(nn.Linear(width, output_dim,bias=bias))

        if use_sigmoid: self.mm.append(nn.Sigmoid())

    def forward(self, x, y=None):
        if y is None: y=x.detach().clone()

        ### calculate x*W*y.T intead of kron(x,y)*vec(W) ###
        # naive way
        x = x@self.first.weight
        x = x@(y.transpose(0,1))

        # einsum implementation
        # x = torch.einsum('ij,wjk,lk->wil',x,self.first.weight,y)
        ##########################################################
        
        x = x.flatten(1,2).transpose(0,1)
        for m in self.mm:
            x = m(x)
        return x
    def name(self):
        return "Kron_MLP"


# MLP with blending matrix in sparse matrix multiplication, not using because it's a bit slow
class Blend_Kron_MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, depth=0, width0=256, width=256, bias=True, use_sigmoid=True):
        super(Blend_Kron_MLP, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        if depth==0: width0 = output_dim
        
        self.mm = nn.ModuleList([])
        self.first = nn.Parameter(2/np.sqrt(width0)*torch.rand(width0, input_dim, input_dim)-1/np.sqrt(width0))
        
        if depth == 1:
            self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, output_dim,bias=bias)))
        if depth > 1:
            self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, width,bias=bias),nn.ReLU(True)))
            for i in range(depth-2):
                self.mm.append(nn.Sequential(nn.Linear(width, width,bias=bias),nn.ReLU(True)))
            self.mm.append(nn.Linear(width, output_dim,bias=bias))

        if use_sigmoid: self.mm.append(nn.Sigmoid())

    def forward(self, B, x, y=None):
        if y is None: y=x.detach().clone()

        ### calculate x*W*y.T intead of kron(x,y)*vec(W) ###
        # naive way
        x = x@self.first
        x = x@(y.transpose(0,1))

        # einsum implementation
        # x = torch.einsum('ij,wjk,lk->wil',x,self.first.weight,y)
        ##########################################################

        x = x.flatten(1,2).transpose(0,1)
        x = B@x
        for m in self.mm:
            x = m(x)
        return x
    def name(self):
        return "Blend_Kron_MLP"



# MLP with blending matrix in indexing and weight sum implementation
class Indexing_Blend_Kron_MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, depth=0, width0=256, width=256, bias=True, use_sigmoid=True):
        super(Indexing_Blend_Kron_MLP, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        if depth==0: width0 = output_dim
        
        self.mm = nn.ModuleList([])
        self.first = nn.Parameter(2/np.sqrt(width0)*torch.rand(width0, input_dim, input_dim)-1/np.sqrt(width0))
        
        if depth == 1:
            self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, output_dim,bias=bias)))
        if depth > 1:
            self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, width,bias=bias),nn.ReLU(True)))
            for i in range(depth-2):
                self.mm.append(nn.Sequential(nn.Linear(width, width,bias=bias),nn.ReLU(True)))
            self.mm.append(nn.Linear(width, output_dim,bias=bias))

        if use_sigmoid: self.mm.append(nn.Sigmoid())

    def forward(self, B, x, y=None):
        if y is None: y=x.detach().clone()

        ### calculate x*W*y.T intead of kron(x,y)*vec(W) and indexing weight sum ###
        # naive way
        x = x@self.first
        x = x@(y.transpose(0,1))
        x = x.flatten(1,2).transpose(0,1)
        x = (x[B[0]]*(B[1].unsqueeze(-1))).sum(1)

        # einsum implementation
        # x = torch.einsum('ij,wjk,lk->wil',x,self.first.weight,y)
        # x = x.flatten(1,2).transpose(0,1)
        # x = torch.einsum('ijk,ij->ik',x[B[0]],B[1])
        ##########################################################

        for m in self.mm:
            x = m(x)
        return x
    def name(self):
        return "Indexing_Blend_Kron_MLP"


# Followings are 3D veisrions
class Kron3_MLP(nn.Module):
    def __init__(self,input_dim=3, output_dim=3, depth=0, width0=256, width=256, bias=True, use_sigmoid=True):
        super(Kron3_MLP, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        if depth==0: width0 = output_dim
        
        self.mm = nn.ModuleList([])
        self.first = nn.Parameter(2/np.sqrt(width0)*torch.rand(width0, input_dim, input_dim, input_dim)-1/np.sqrt(width0))

        if depth == 1:
            self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, output_dim, bias=bias)))
        if depth > 1:
            self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, width, bias=bias),nn.ReLU(True)))
            for i in range(depth-2):
                self.mm.append(nn.Sequential(nn.Linear(width, width, bias=bias),nn.ReLU(True)))
            self.mm.append(nn.Linear(width, output_dim, bias=bias))

        if use_sigmoid: self.mm.append(nn.Sigmoid())
    def forward(self, x, y=None,z=None):
        if y is None: y=x.detach().clone()
        if z is None: z=x.detach().clone()
        
        ### calculate mode-n multiplication intead of kron(x,y,z)*vec(W) ###
        # naive way
        x = x@self.first
        x = x@(y.transpose(0,1))
        x = z@(x.transpose(1,2))
        
        # einsum implementation
        # x = torch.einsum('ai,bj,ck,wijk->wabc',x,y,z,self.first.weight)
        ##########################################################

        x = x.flatten(1,3).transpose(0,1)
        for m in self.mm:
            x = m(x)
        return x
    def name(self):
        return "Kron3_MLP"


# class Blend_Kron3_MLP(nn.Module):
#     def __init__(self,input_dim=2, output_dim=3, depth=0,width0=256,width=256, use_sigmoid=True):
#         super(Blend_Kron3_MLP, self).__init__()
#         self.use_sigmoid = use_sigmoid
        
#         if depth==0: width0 = output_dim
        
#         self.mm = nn.ModuleList([])
#         self.first = nn.ParameterDict({
#                 'weight': nn.Parameter(2/np.sqrt(width0)*torch.rand(width0, input_dim, input_dim, input_dim)-1/np.sqrt(width0))})
#         if depth == 1:
#             self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, output_dim)))
#         if depth > 1:
#             self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, width),nn.ReLU(True)))
#             for i in range(depth-1):
#                 self.mm.append(nn.Sequential(nn.Linear(width, width),nn.ReLU(True)))
#             self.mm.append(nn.Linear(width, output_dim))
#         if use_sigmoid: self.mm.append(nn.Sigmoid())
#     def forward(self, B, x, y=None,z=None):
#         if y is None: y=x.detach().clone()
#         if z is None: z=x.detach().clone()
#         x = x@self.first.weight
#         x = x@(y.transpose(0,1))
#         x = z@(x.transpose(1,2))
#         x = x.flatten(1,3).transpose(0,1)
#         x = B@x
#         for m in self.mm:
#             x = m(x)
#         return x
#     def name(self):
#         return "Blend_Kron3_MLP" 



class Indexing_Blend_Kron3_MLP(nn.Module):
    def __init__(self,input_dim=3, output_dim=3, depth=0, width0=256, width=256, bias=True, use_sigmoid=True):
        super(Indexing_Blend_Kron3_MLP, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        if depth==0: width0 = output_dim
        
        self.mm = nn.ModuleList([])
        self.first = nn.Parameter(2/np.sqrt(width0)*torch.rand(width0, input_dim, input_dim, input_dim)-1/np.sqrt(width0))

        if depth == 1:
            self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, output_dim, bias=bias)))
        if depth > 1:
            self.mm.append(nn.Sequential(nn.ReLU(True),nn.Linear(width0, width, bias=bias),nn.ReLU(True)))
            for i in range(depth-2):
                self.mm.append(nn.Sequential(nn.Linear(width, width, bias=bias),nn.ReLU(True)))
            self.mm.append(nn.Linear(width, output_dim, bias=bias))

        if use_sigmoid: self.mm.append(nn.Sigmoid())
    def forward(self, B, x, y=None,z=None):
        if y is None: y=x.detach().clone()
        if z is None: z=x.detach().clone()

        ### calculate mode-n multiplication intead of kron(x,y,z)*vec(W) ###
        # naive way
        x = x@self.first
        x = x@(y.transpose(0,1))
        x = z@(x.transpose(1,2))
        x = x.flatten(1,3).transpose(0,1)
        x = (x[B[0]]*(B[1].unsqueeze(-1))).sum(1)
        
        # einsum implementation
        # x = torch.einsum('ai,bj,ck,wijk->wabc',x,y,z,self.first.weight)
        # x = x.flatten(1,3).transpose(0,1)
        # x = torch.einsum('ijk,ij->ik',x[B[0]],B[1])
        ##########################################################
        for m in self.mm:
            x = m(x)
        return x
    def name(self):
        return "Indexing_Blend_Kron3_MLP"         
