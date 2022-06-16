import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# encoding funcs for 1D, 2D and 3D
class encoding_func_1D:
    def __init__(self, name, param=None):
        # param should be in form of [sigma,feature_dimension]
        self.name = name
        
        if name == 'none': self.dim=1
        elif name == 'basic': self.dim=2
        else:
            self.dim = param[1]
            if name == 'RFF':
                self.sig = param[0]
                self.b = param[0]*torch.randn((int(param[1]/2),1))
            elif name == 'rffb':
                self.b = param[0]
            elif name == 'LinF':
                self.b = torch.linspace(2.**0., 2.**param[0], steps=int(param[1]/2)).reshape(-1,1)
            elif name == 'LogF':
                self.b = 2.**torch.linspace(0., param[0], steps=int(param[1]/2)).reshape(-1,1)
            elif name == 'Gau':
                self.dic = torch.linspace(0., 1, steps=param[1]+1)[:-1].reshape(1,-1)
                self.sig = param[0]
            elif name == 'Tri':
                self.dic = torch.linspace(0., 1, steps=param[1]+1)[:-1].reshape(1,-1)
                if param[0] is None: self.d = 1/param[1]
                else: self.d = param[0]
            else:
                print('Undifined encoding!')
    def __call__(self, x):
        if self.name == 'none':
            return x
        elif self.name == 'basic':
            emb = torch.cat((torch.sin((2.*np.pi*x)),torch.cos((2.*np.pi*x))),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'RFF')|(self.name == 'rffb')|(self.name == 'LinF')|(self.name == 'LogF'):
            emb = torch.cat((torch.sin((2.*np.pi*x) @ self.b.T),torch.cos((2.*np.pi*x) @ self.b.T)),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif self.name == 'Gau':
            emb = (-0.5*(x-self.dic)**2/(self.sig**2)).exp()
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif self.name == 'Tri':
            emb = (1-(x-self.dic).abs()/self.d)
            emb = emb*(emb>0)
            emb = emb/(emb.norm(dim=1).max())
            return emb


# simple 2D encoding. For Fourier based methods we use 2 directions. For shifted verision encoder we use 4 directions. The total feature dimension is same. 
class encoding_func_2D:
    def __init__(self, name, param=None):
        self.name = name

        if name == 'none': self.dim=2
        elif name == 'basic': self.dim=4
        else:
            self.dim = param[1]
            if name == 'RFF':
                self.b = param[0]*torch.randn((int(param[1]/2),2))
            elif name == 'rffb':
                self.b = param[0]
            elif name == 'LinF':
                self.b = torch.linspace(2.**0., 2.**param[0], steps=int(param[1]/4)).reshape(-1,1)
            elif name == 'LogF':
                self.b = 2.**torch.linspace(0., param[0], steps=int(param[1]/4)).reshape(-1,1)
            elif name == 'Gau2':
                self.dic = torch.linspace(0., 1, steps=int(param[1]/2)+1)[:-1].reshape(1,-1)
                self.sig = param[0]
            elif (name == 'Gau4')|(name == 'Gau'):
                self.dic = torch.linspace(0., 1, steps=int(param[1]/4)+1)[:-1].reshape(1,-1)
                self.sig = param[0]
            elif name == 'Tri2':
                self.dic = torch.linspace(0., 1, steps=int(param[1]/2)+1)[:-1].reshape(1,-1)
                if param[0] is None: self.d = 1/param[1]
                else: self.d = param[0]
            elif (name == 'Tri4')|(name == 'Tri'):
                self.dic = torch.linspace(0., 1, steps=int(param[1]/4)+1)[:-1].reshape(1,-1)
                if param[0] is None: self.d = 1/param[1]
                else: self.d = param[0]
            else:
                print('Undifined encoding!')
    def __call__(self, x):
        if self.name == 'none':
            return x
        elif self.name == 'basic':
            emb = torch.cat((torch.sin((2.*np.pi*x)),torch.cos((2.*np.pi*x))),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'RFF')|(self.name == 'rffb'):
            emb = torch.cat((torch.sin((2.*np.pi*x) @ self.b.T),torch.cos((2.*np.pi*x) @ self.b.T)),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'LinF')|(self.name == 'LogF'):
            emb1 = torch.cat((torch.sin((2.*np.pi*x[:,:1]) @ self.b.T),torch.cos((2.*np.pi*x[:,:1]) @ self.b.T)),1)
            emb2 = torch.cat((torch.sin((2.*np.pi*x[:,1:2]) @ self.b.T),torch.cos((2.*np.pi*x[:,1:2]) @ self.b.T)),1)
            emb = torch.cat([emb1,emb2],1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif self.name == 'Gau2':
            emb1 = (-0.5*(x[:,:1]-self.dic)**2/(self.sig**2)).exp()
            emb2 = (-0.5*(x[:,1:2]-self.dic)**2/(self.sig**2)).exp()
            emb = torch.cat([emb1,emb2],1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'Gau4')|(self.name == 'Gau'):
            emb1 = (-0.5*(x[:,:1]-self.dic)**2/(self.sig**2)).exp()
            emb2 = (-0.5*(x[:,1:2]-self.dic)**2/(self.sig**2)).exp()
            emb3 = (-0.5*(0.5*(x[:,:1]+x[:,1:2])-self.dic)**2/(self.sig**2)).exp()
            emb4 = (-0.5*(0.5*(x[:,:1]-x[:,1:2]+1)-self.dic)**2/(self.sig**2)).exp()
            emb = torch.cat([emb1,emb2,emb3,emb4],1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif self.name == 'Tri2':
            emb1 = (1-(x[:,:1]-self.dic).abs()/self.d)
            emb1 = emb1*(emb1>0)
            emb2 = (1-(x[:,1:2]-self.dic).abs()/self.d)
            emb2 = emb2*(emb2>0)
            emb = torch.cat([emb1,emb2],1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'Tri4')|(self.name == 'Tri'):
            emb1 = (1-(x[:,:1]-self.dic).abs()/self.d)
            emb1 = emb1*(emb1>0)
            emb2 = (1-(x[:,1:2]-self.dic).abs()/self.d)
            emb2 = emb2*(emb2>0)
            emb3 = (1-(0.5*(x[:,:1]+x[:,1:2])-self.dic).abs()/self.d)
            emb3 = emb3*(emb3>0)
            emb4 = (1-(0.5*(x[:,:1]-x[:,1:2]+1)-self.dic).abs()/self.d)
            emb4 = emb4*(emb4>0)
            emb = torch.cat([emb1,emb2,emb3,emb4],1)
            emb = emb/(emb.norm(dim=1).max())
            return emb


# simple 3D encoding. All method use 3 directions.
class encoding_func_3D:
    def __init__(self, name, param=None):
        self.name = name

        if name == 'none': self.dim=2
        elif name == 'basic': self.dim=4
        else:
            self.dim = param[1]
            if name == 'RFF':
                self.b = param[0]*torch.randn((int(param[1]/2),3))
            elif name == 'rffb':
                self.b = param[0]
            elif name == 'LinF':
                self.b = torch.linspace(2.**0., 2.**param[0], steps=int(param[1]/6)).reshape(-1,1)
            elif name == 'LogF':
                self.b = 2.**torch.linspace(0., param[0], steps=int(param[1]/6)).reshape(-1,1)
            elif name == 'Gau':
                self.dic = torch.linspace(0., 1, steps=int(param[1]/3)+1)[:-1].reshape(1,-1)
                self.sig = param[0]
            elif name == 'Tri':
                self.dic = torch.linspace(0., 1, steps=int(param[1]/3)+1)[:-1].reshape(1,-1)
                if param[0] is None: self.d = 1/param[1]
                else: self.d = param[0]
            else:
                print('Undifined encoding!')
    def __call__(self, x):
        if self.name == 'none':
            return x
        elif self.name == 'basic':
            emb = torch.cat((torch.sin((2.*np.pi*x)),torch.cos((2.*np.pi*x))),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'RFF')|(self.name == 'rffb'):
            emb = torch.cat((torch.sin((2.*np.pi*x) @ self.b.T),torch.cos((2.*np.pi*x) @ self.b.T)),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'LinF')|(self.name == 'LogF'):
            emb1 = torch.cat((torch.sin((2.*np.pi*x[:,:1]) @ self.b.T),torch.cos((2.*np.pi*x[:,:1]) @ self.b.T)),1)
            emb2 = torch.cat((torch.sin((2.*np.pi*x[:,1:2]) @ self.b.T),torch.cos((2.*np.pi*x[:,1:2]) @ self.b.T)),1)
            emb3 = torch.cat((torch.sin((2.*np.pi*x[:,2:3]) @ self.b.T),torch.cos((2.*np.pi*x[:,2:3]) @ self.b.T)),1)
            emb = torch.cat([emb1,emb2,emb3],1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif self.name == 'Gau':
            emb1 = (-0.5*(x[:,:1]-self.dic)**2/(self.sig**2)).exp()
            emb2 = (-0.5*(x[:,1:2]-self.dic)**2/(self.sig**2)).exp()
            emb3 = (-0.5*(x[:,2:3]-self.dic)**2/(self.sig**2)).exp()
            emb = torch.cat([emb1,emb2,emb3],1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif self.name == 'Tri':
            emb1 = (1-(x[:,:1]-self.dic).abs()/self.d)
            emb1 = emb1*(emb1>0)
            emb2 = (1-(x[:,1:2]-self.dic).abs()/self.d)
            emb2 = emb2*(emb2>0)
            emb3 = (1-(x[:,2:3]-self.dic).abs()/self.d)
            emb3 = emb3*(emb3>0)
            emb = torch.cat([emb1,emb2,emb3],1)
            emb = emb/(emb.norm(dim=1).max())
            return emb


# blending matrix for 2D random samples
class blending_func_2D:
    '''
    encoding_func : inner encoding func. The name will be used to choose corresponding closed form distance func, which will be much faster than calculating experimentally.
    dim : number of inner encoing func to inerpolate, usually equals to the feature dimension of inner encoding
    indexing: defalt as True to return grid point index and weights. Set False to return a sparse marix but will be sloer in the future.
    '''
    def __init__(self, encoding_func, dim=256,indexing=True):
        self.name = encoding_func.name
        self.indexing = indexing

        if dim is None: self.dim = encoding_func.dim
        else: self.dim = dim
        
        if self.name == 'RFF':
            self.D = lambda x1,x2: (-2*(np.pi*(x1-x2)/(self.dim-1)*encoding_func.sig)**2).exp()
        elif self.name == 'Gau':
            self.D = lambda x1,x2: (-0.25*((x1-x2)/(self.dim-1))**2/(encoding_func.sig**2)).exp()
        elif self.name == 'Tri':
            self.D = lambda x1,x2: 0.25*torch.maximum(2*encoding_func.d-(x1-x2).abs()/(self.dim-1),torch.tensor(0))**2
        else:
            self.D = lambda x1,x2: (encoding_func(x1/(self.dim-1))*encoding_func(x2/(self.dim-1))).sum(-1).unsqueeze(1)

    def __call__(self, x):
        # make x in the grid    
        x = x.clamp(0,1-1e-3)
        x = (self.dim-1)*x
        y = x[:,1:2]
        x = x[:,:1]
        xmin = torch.floor(x)
        ymin = torch.floor(y)


        xd0 = self.D(xmin,xmin)
        xdd = self.D(xmin,xmin+1)
        xd1 = self.D(xmin+1,xmin+1)
        xda = self.D(xmin,x)
        xdb = self.D(xmin+1,x)
        xff = xd0*xd1-xdd**2

        xa = (xda*xd1-xdb*xdd)/xff
        xb = (xdb*xd0-xda*xdd)/xff

        yd0 = self.D(ymin,ymin)
        ydd = self.D(ymin,ymin+1)
        yd1 = self.D(ymin+1,ymin+1)
        yda = self.D(ymin,y)
        ydb = self.D(ymin+1,y)
        yff = yd0*yd1-ydd**2

        ya = (yda*yd1-ydb*ydd)/yff
        yb = (ydb*yd0-yda*ydd)/yff

        xs = xa+xb
        xa = xa/xs
        xb = xb/xs
        
        ys = ya+yb
        ya = ya/ys
        yb = yb/ys


        if self.indexing:
            return [torch.cat([xmin*self.dim+ymin,xmin*self.dim+ymin+1,(xmin+1)*self.dim+ymin,(xmin+1)*self.dim+ymin+1],1).type(torch.LongTensor),torch.cat([xa*ya,xa*yb,xb*ya,xb*yb],1)]
        else:
            c = torch.cat([xa*ya,xa*yb,xb*ya,xb*yb],0)

            y = torch.cat([xmin*self.dim+ymin,xmin*self.dim+ymin+1,(xmin+1)*self.dim+ymin,(xmin+1)*self.dim+ymin+1],0).type(torch.IntTensor)
            x = torch.linspace(0,x.shape[0]-1,x.shape[0],dtype=int).reshape(-1,1).repeat(4,1)
            return torch.sparse_coo_tensor(torch.cat([x,y],1).T, c.reshape(-1), (int(x.shape[0]/4), self.dim**2))





class blending_func_3D:
    '''
    encoding_func : inner encoding func. The name will be used to choose corresponding closed form distance func, which will be much faster than calculating experimentally.
    dim : number of inner encoing func to inerpolate, usually equals to the feature dimension of inner encoding
    indexing: defalt as True to return grid point index and weights. Set False to return a sparse marix but will be sloer in the future.
    '''
    def __init__(self, encoding_func, dim=None,indexing=True):
        self.name = encoding_func.name
        self.indexing = indexing

        if dim is None: self.dim = encoding_func.dim
        else: self.dim = dim
        
        if self.name == 'RFF':
            self.D = lambda x1,x2: (-2*(np.pi*(x1-x2)/(self.dim-1)*encoding_func.sig)**2).exp()
        elif self.name == 'Gau':
            self.D = lambda x1,x2: (-0.25*((x1-x2)/(self.dim-1))**2/(encoding_func.sig**2)).exp()
        elif self.name == 'Tri':
            self.D = lambda x1,x2: 0.25*torch.maximum(2*encoding_func.d-(x1-x2).abs()/(self.dim-1),torch.tensor(0))**2
        else:
            self.D = lambda x1,x2: (encoding_func(x1/(self.dim-1))*encoding_func(x2/(self.dim-1))).sum(-1).unsqueeze(1)

    def __call__(self, x):
        # make x in the grid    
        x = x.clamp(0,1-1e-3)
        x = (self.dim-1)*x
        y = x[:,1:2]
        z = x[:,2:3]
        x = x[:,:1]
        xmin = torch.floor(x)
        ymin = torch.floor(y)
        zmin = torch.floor(z)

        if self.name=='RFF' or self.name=='Gau' or self.name=='Tri':
            d0 = self.D(torch.tensor([0]),torch.tensor([0]))
            dd = self.D(torch.tensor([0]),torch.tensor([1]))
            ff = d0**2-dd**2

            xda = self.D(xmin,x)
            xdb = self.D(xmin+1,x)

            xa = (xda*d0-xdb*dd)/ff
            xb = (xdb*d0-xda*dd)/ff

            yda = self.D(ymin,y)
            ydb = self.D(ymin+1,y)

            ya = (yda*d0-ydb*dd)/ff
            yb = (ydb*d0-yda*dd)/ff

            zda = self.D(zmin,z)
            zdb = self.D(zmin+1,z)

            za = (zda*d0-zdb*dd)/ff
            zb = (zdb*d0-zda*dd)/ff

            xs = xa+xb
            xa = xa/xs
            xb = xb/xs
            
            ys = ya+yb
            ya = ya/ys
            yb = yb/ys
            
            zs = za+zb
            za = za/zs
            zb = zb/zs

            N=self.dim
            Ns = x.shape[0]
            if self.indexing:
                c = torch.cat([xa*ya*za,xa*ya*zb,xa*yb*za,xa*yb*zb,xb*ya*za,xb*ya*zb,xb*yb*za,xb*yb*zb],1)
                y = torch.cat([xmin*N**2+ymin*N+zmin,xmin*N**2+ymin*N+zmin+1,xmin*N**2+(ymin+1)*N+zmin,xmin*N**2+(ymin+1)*N+zmin+1,
                                    (xmin+1)*N**2+ymin*N+zmin,(xmin+1)*N**2+ymin*N+zmin+1,(xmin+1)*N**2+(ymin+1)*N+zmin,(xmin+1)*N**2+(ymin+1)*N+zmin+1],1).type(torch.LongTensor)
                return [y,c]
            else:
                c = torch.cat([xa*ya*za,xa*ya*zb,xa*yb*za,xa*yb*zb,xb*ya*za,xb*ya*zb,xb*yb*za,xb*yb*zb],0)

                
                y = torch.cat([xmin*N**2+ymin*N+zmin,xmin*N**2+ymin*N+zmin+1,xmin*N**2+(ymin+1)*N+zmin,xmin*N**2+(ymin+1)*N+zmin+1,
                                    (xmin+1)*N**2+ymin*N+zmin,(xmin+1)*N**2+ymin*N+zmin+1,(xmin+1)*N**2+(ymin+1)*N+zmin,(xmin+1)*N**2+(ymin+1)*N+zmin+1],0).type(torch.IntTensor)
                x = torch.range(0,Ns-1,dtype=int).reshape(-1,1).repeat(8,1)
                return torch.sparse_coo_tensor(torch.cat([x,y],1).T, c.reshape(-1), (Ns, self.dim**3))

        else:
            xd0 = self.D(xmin,xmin)
            xdd = self.D(xmin,xmin+1)
            xd1 = self.D(xmin+1,xmin+1)
            xda = self.D(xmin,x)
            xdb = self.D(xmin+1,x)
            xff = xd0*xd1-xdd**2

            xa = (xda*xd1-xdb*xdd)/xff
            xb = (xdb*xd0-xda*xdd)/xff

            yd0 = self.D(ymin,ymin)
            ydd = self.D(ymin,ymin+1)
            yd1 = self.D(ymin+1,ymin+1)
            yda = self.D(ymin,y)
            ydb = self.D(ymin+1,y)
            yff = yd0*yd1-ydd**2

            ya = (yda*yd1-ydb*ydd)/yff
            yb = (ydb*yd0-yda*ydd)/yff


            zd0 = self.D(zmin,zmin)
            zdd = self.D(zmin,zmin+1)
            zd1 = self.D(zmin+1,zmin+1)
            zda = self.D(zmin,z)
            zdb = self.D(zmin+1,z)
            zff = zd0*zd1-zdd**2

            za = (zda*zd1-zdb*zdd)/zff
            zb = (zdb*zd0-zda*zdd)/zff

            xs = xa+xb
            xa = xa/xs
            xb = xb/xs
            
            ys = ya+yb
            ya = ya/ys
            yb = yb/ys
            
            zs = za+zb
            za = za/zs
            zb = zb/zs

            N=self.dim
            Ns = x.shape[0]
            if self.indexing:
                c = torch.cat([xa*ya*za,xa*ya*zb,xa*yb*za,xa*yb*zb,xb*ya*za,xb*ya*zb,xb*yb*za,xb*yb*zb],1)
                y = torch.cat([xmin*N**2+ymin*N+zmin,xmin*N**2+ymin*N+zmin+1,xmin*N**2+(ymin+1)*N+zmin,xmin*N**2+(ymin+1)*N+zmin+1,
                                    (xmin+1)*N**2+ymin*N+zmin,(xmin+1)*N**2+ymin*N+zmin+1,(xmin+1)*N**2+(ymin+1)*N+zmin,(xmin+1)*N**2+(ymin+1)*N+zmin+1],1).type(torch.LongTensor)
                return [y,c]
            else:
                c = torch.cat([xa*ya*za,xa*ya*zb,xa*yb*za,xa*yb*zb,xb*ya*za,xb*ya*zb,xb*yb*za,xb*yb*zb],0)

                
                y = torch.cat([xmin*N**2+ymin*N+zmin,xmin*N**2+ymin*N+zmin+1,xmin*N**2+(ymin+1)*N+zmin,xmin*N**2+(ymin+1)*N+zmin+1,
                                    (xmin+1)*N**2+ymin*N+zmin,(xmin+1)*N**2+ymin*N+zmin+1,(xmin+1)*N**2+(ymin+1)*N+zmin,(xmin+1)*N**2+(ymin+1)*N+zmin+1],0).type(torch.IntTensor)
                x = torch.range(0,Ns-1,dtype=int).reshape(-1,1).repeat(8,1)
                return torch.sparse_coo_tensor(torch.cat([x,y],1).T, c.reshape(-1), (Ns, self.dim**3))



class fast_blending_func_3D:
    '''
    A fast version (may not really) without many options for common use.
    encoding_func : inner encoding func. The name will be used to choose corresponding closed form distance func, which will be much faster than calculating experimentally.
    dim : number of inner encoing func to inerpolate, usually equals to the feature dimension of inner encoding
    '''
    def __init__(self, encoding_func, dim=None):
        self.name = encoding_func.name

        if dim is None: self.dim = encoding_func.dim
        else: self.dim = dim
        
        if self.name == 'RFF':
            self.D = lambda t: (-2*(np.pi*(t)/(self.dim-1)*encoding_func.sig)**2).exp()
        elif self.name == 'Gau':
            self.D = lambda t: (-0.25*((t)/(self.dim-1))**2/(encoding_func.sig**2)).exp()
        elif self.name == 'Tri':
            self.D = lambda t: 0.25*torch.maximum(2*encoding_func.d-(t).abs()/(self.dim-1),torch.tensor(0))**2
        else:
            print('No fast version!')

        # constant coefficients
        d0 = self.D(torch.tensor([0]))
        dd = self.D(torch.tensor([1]))
        ff = d0**2-dd**2
        self.coe = torch.tensor([[d0,-dd],[-dd,d0]])/ff

        self.dim2 = self.dim**2
        self.idx_matrix = torch.tensor([self.dim**2,self.dim,1.0]).reshape(3,1)

    def __call__(self, x):
        # make x in the grid    
        x = x.clamp(0,1-1e-3)
        x = (self.dim-1)*x
        xmin = torch.floor(x)
        
        d_ratio = torch.stack([self.D(xmin-x),self.D(xmin+1-x)],-1)
        itp_weights = d_ratio@self.coe

        # calcualte mixing product
        xy = itp_weights[:,0,0]*itp_weights[:,1,0]
        xy1 = itp_weights[:,0,0]*itp_weights[:,1,1]
        x1y = itp_weights[:,0,1]*itp_weights[:,1,0]
        x1y1 = itp_weights[:,0,1]*itp_weights[:,1,1]
        z = itp_weights[:,2,0]
        z1 = itp_weights[:,2,1]
        c = torch.stack([xy*z,xy*z1,xy1*z,xy1*z1,x1y*z,x1y*z1,x1y1*z,x1y1*z1],1)

        x_idx = xmin@self.idx_matrix
        y = torch.cat([x_idx,x_idx+1],1)
        y = torch.cat([y,y+self.dim],1)
        y = torch.cat([y,y+self.dim2],1)
        return [y.type(torch.LongTensor),c]



def srank_func(X,return_l1=False):
    (_,s,_) = torch.svd(X)
    sr2 = (s*s).sum()/s[0]/s[0]
    sr1 = s.sum()/s[0]
    if return_l1:
        return sr1,sr2
    else:
        return sr2

    
def psnr_func(x,y,return_mse=False):
    diff = x - y
    mse = (diff*diff).flatten().mean()
    if return_mse:
        return -10*(mse.log10()),mse
    else:
        return -10*(mse.log10())


def smooth(X,filter):
    sx = (torch.nn.functional.conv1d(X.unsqueeze(1),filter)**2).mean()
    sy = (torch.nn.functional.conv1d(X.T.unsqueeze(1),filter)**2).mean()
    return sx+sy


def smooth_3D(X,filter):
    sx = (torch.nn.functional.conv1d(X.flatten(0,1).unsqueeze(1),filter)**2).mean()
    sy = (torch.nn.functional.conv1d(X.transpose(1,2).flatten(0,1).unsqueeze(1),filter)**2).mean()
    sz = (torch.nn.functional.conv1d(X.transpose(0,2).flatten(0,1).unsqueeze(1),filter)**2).mean()
    return sx+sy+sz
