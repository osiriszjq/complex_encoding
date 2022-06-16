import numpy as np
import time, torch, logging
from MLPs import *
from utils import *
            

def train_simple_2D(signals,encoding_func,epochs=2000,lr=1e-2,criterion=nn.MSELoss(),device="cpu",
                depth=4,width=256,bias=True,use_sigmoid=False,N_repeat=1,test_mode='rest',logger=None):

    '''
    Trainer of simple encoding for 2D image reconstruction with separable coordinates.

    The input args are four parts:

    signals: a pytorch tensor shape in N_image x sample_N x sample_N x 3
    encoding_func: encoding function
    
    epochs,lr,criterion are optimization settings

    depth,width,bias,use_sigmoid are network cofiguration

    N_repeat: number of repeat times for each example
    test_mode: default as 'rest' to use all points except training ones for test. 'mid' will use a shift grid of training points. 'all' will use all ponts including training ones.
    logger: if is None, loggers will print out.

    Returns:

    running time: N_test x N_repeat
    train psnr: N_test x N_repeat
    test psnr: N_test x N_repeat
    reconstuction results: same size as input signals
    '''

    
    sample_N =signals.shape[1]
    N_test = signals.shape[0]

    trn_psnr_ = np.zeros((N_test,N_repeat))
    tst_psnr_ = np.zeros((N_test,N_repeat))
    time_ = np.zeros((N_test,N_repeat))
    rec_ = np.zeros((N_test,sample_N,sample_N,3))

    # build the meshgrid for coordinates
    x1 = np.linspace(0, 1, sample_N+1)[:-1]
    all_data = np.stack(np.meshgrid(x1,x1), axis=-1)
    

    for i in range(N_test):
        # Prepare the targets
        all_target = signals[i].squeeze()
        train_label = all_target[::2,::2].reshape(-1,3).type(torch.FloatTensor)
        if test_mode == 'mid':
            test_label = all_target[1::2,1::2].reshape(-1,3).type(torch.FloatTensor)
        elif test_mode == 'rest':
            test_label1 = all_target[1::2,::2].reshape(-1,3).type(torch.FloatTensor)
            test_label2 = all_target[1::2,1::2].reshape(-1,3).type(torch.FloatTensor)
            test_label3 = all_target[::2,1::2].reshape(-1,3).type(torch.FloatTensor)
            test_label = torch.cat([test_label1,test_label2,test_label3],0)
        elif test_mode == 'all':
            test_label = all_target.reshape(-1,3).type(torch.FloatTensor)
        

        for k in range(N_repeat):
            start_time = time.time()
            train_data = encoding_func(torch.from_numpy(all_data[::2,::2].reshape(-1,2)).type(torch.FloatTensor))
            

            train_data, train_label = train_data.to(device),train_label.to(device)

            # regular MLP
            model = MLP(input_dim=encoding_func.dim,output_dim=3,depth=depth,width=width,bias=bias,use_sigmoid=use_sigmoid).to(device)
            # Set the optimization
            optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999),weight_decay=1e-8)
            
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                out = model(train_data)
                loss = criterion(out, train_label)

                loss.backward()
                optimizer.step()

                
            time_[i,k] = time.time() - start_time


            if test_mode == 'mid':
                test_data = encoding_func(torch.from_numpy(all_data[1::2,1::2].reshape(-1,2)).type(torch.FloatTensor))
            elif test_mode == 'rest':
                test_data1 = encoding_func(torch.from_numpy(all_data[1::2,::2].reshape(-1,2)).type(torch.FloatTensor))
                test_data2 = encoding_func(torch.from_numpy(all_data[1::2,1::2].reshape(-1,2)).type(torch.FloatTensor))
                test_data3 = encoding_func(torch.from_numpy(all_data[::2,1::2].reshape(-1,2)).type(torch.FloatTensor))
                test_data = torch.cat([test_data1,test_data2,test_data3],0)
            elif test_mode == 'all':
                test_data = encoding_func(torch.from_numpy(all_data.reshape(-1,2)).type(torch.FloatTensor))  
            else:
                print('Wrong test_mode!')
                return -1


            test_data, test_label = test_data.to(device),test_label.to(device)

            model.eval()
            with torch.no_grad():
                trn_psnr_[i,k] = psnr_func(model(train_data),train_label)
                tst_psnr_[i,k] = psnr_func(model(test_data),test_label)
            if logger == None:
                print("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
            else:
                logger.info("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
            
        rec_[i] = model(encoding_func(torch.from_numpy(all_data.reshape(-1,2)).type(torch.FloatTensor)).to(device)).reshape(sample_N,sample_N,3).detach().cpu()
    return time_, trn_psnr_,tst_psnr_,rec_




def train_kron_2D(signals,encoding_func,epochs=2000,lr=1e-1,criterion=nn.MSELoss(),device="cpu",
                depth=0,width=256,width0=256,bias=True,use_sigmoid=False,N_repeat=1,test_mode='rest',logger=None):

    '''
    Trainer of complex encoding for 2D image reconstruction with separable coordinates.

    The input args are four parts:

    signals: a pytorch tensor shape in N_image x sample_N x sample_N x 3
    encoding_func: encoding function
    
    epochs,lr,criterion are optimization settings

    depth,width,width0,bias,use_sigmoid are network cofiguration

    N_repeat: number of repeat times for each example
    test_mode: default as 'rest' to use all points except training ones for test. 'mid' will use a shift grid of training points. 'all' will use all ponts including training ones.
    logger: if is None, loggers will print out.

    Returns:

    running time: N_test x N_repeat
    train psnr: N_test x N_repeat
    test psnr: N_test x N_repeat
    reconstuction results: same size as input signals
    '''

    
    sample_N =signals.shape[1]
    N_test = signals.shape[0]

    trn_psnr_ = np.zeros((N_test,N_repeat))
    tst_psnr_ = np.zeros((N_test,N_repeat))
    time_ = np.zeros((N_test,N_repeat))
    rec_ = np.zeros((N_test,sample_N,sample_N,3))

    # Here we only use 1D grids
    all_data = np.linspace(0, 1, sample_N+1)[:-1]
    

    for i in range(N_test):
        # Prepare the targets
        all_target = signals[i].squeeze()
        train_label = all_target[::2,::2].reshape(-1,3).type(torch.FloatTensor).to(device)
        if test_mode == 'mid':
            test_label = all_target[1::2,::2].reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[True]],device=device).repeat(int(sample_N/2),int(sample_N/2)).reshape(-1)
        elif test_mode == 'rest':
            test_label = all_target.reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[False,True],[True,True]],device=device).repeat(int(sample_N/2),int(sample_N/2)).reshape(-1)
        elif test_mode == 'all':
            test_label = all_target.reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[True]],device=device).repeat(sample_N,sample_N).reshape(-1)
          
        
        for k in range(N_repeat):
            start_time = time.time()
            train_data = torch.from_numpy(all_data[::2].reshape(-1,1)).type(torch.FloatTensor)
            
            # Initialize classification model to learn
            model = Kron_MLP(input_dim=encoding_func.dim,output_dim=3,depth=depth,width0=width0,width=width,bias=bias,use_sigmoid=use_sigmoid).to(device)
            # Set the optimization
            optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999),weight_decay=1e-8)

            train_data = encoding_func(train_data).to(device)
                
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                out = model(train_data)
                loss = criterion(out, train_label)

                loss.backward()
                optimizer.step()


            time_[i,k] = time.time() - start_time
            model.eval()

            if test_mode == 'mid':
                test_data = torch.from_numpy(all_data[1::2].reshape(-1,1)).type(torch.FloatTensor)
            elif test_mode == 'rest':
                test_data = torch.from_numpy(all_data.reshape(-1,1)).type(torch.FloatTensor)
            elif test_mode == 'all':
                test_data = torch.from_numpy(all_data.reshape(-1,1)).type(torch.FloatTensor)
            else:
                print('Wrong test_mode!')
                return -1


            test_data = encoding_func(test_data).to(device)

            with torch.no_grad():
                trn_psnr_[i,k] = psnr_func(model(train_data),train_label)
                tst_psnr_[i,k] = psnr_func(model(test_data)[test_mask],test_label[test_mask])
        
            if logger == None:
                print("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
            else:
                logger.info("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
        rec_[i] = model(encoding_func(torch.from_numpy(all_data.reshape(-1,1)).type(torch.FloatTensor)).to(device)).reshape(sample_N,sample_N,3).detach().cpu()
    return time_,trn_psnr_,tst_psnr_,rec_




def train_closed_form_2D(signals,encoding_func,device="cpu",test_mode='rest',N_repeat=1,logger=None):

    '''
    Closed form solution of complex encoding for 2D image reconstruction with separable coordinates. No training.

    The input args are two parts:

    signals: a pytorch tensor shape in N_image x sample_N x sample_N x 3
    encoding_func: encoding function

    N_repeat: number of repeat times for each example
    test_mode: default as 'rest' to use all points except training ones for test. 'mid' will use a shift grid of training points. 'all' will use all ponts including training ones.
    logger: if is None, loggers will print out.

    Returns:

    running time: N_test x N_repeat
    train psnr: N_test x N_repeat
    test psnr: N_test x N_repeat
    reconstuction results: same size as input signals
    '''

    
    sample_N =signals.shape[1]
    N_test = signals.shape[0]

    trn_psnr_ = np.zeros((N_test,N_repeat))
    tst_psnr_ = np.zeros((N_test,N_repeat))
    time_ = np.zeros((N_test,N_repeat))
    rec_ = np.zeros((N_test,sample_N,sample_N,3))

    # Here we only use 1D grids
    all_data = np.linspace(0, 1, sample_N+1)[:-1]
    
    
    for i in range(N_test):
        # Prepare the targets
        all_target = signals[i].squeeze()
        train_label = all_target[::2,::2].type(torch.FloatTensor).to(device)
        if test_mode == 'mid':
            test_label = all_target[1::2,1::2].reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[True]],device=device).repeat(int(sample_N/2),int(sample_N/2)).reshape(-1)
        elif test_mode == 'rest':
            test_label = all_target.reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[False,True],[True,True]],device=device).repeat(int(sample_N/2),int(sample_N/2)).reshape(-1)
        elif test_mode == 'all':
            test_label = all_target.reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[True]],device=device).repeat(sample_N,sample_N).reshape(-1)
          
        
        for k in range(N_repeat):
            start_time = time.time()
            train_data = torch.from_numpy(all_data[::2].reshape(-1,1)).type(torch.FloatTensor)

            train_data = encoding_func(train_data).to(device)

            # two ways to calculate the inverse of a matrix
            #ix = torch.linalg.pinv(train_data)
            ix = torch.linalg.lstsq(train_data, torch.eye(train_data.shape[0]).to(device)).solution
            W= ix@(train_label.transpose(0,2))@ix.T
            
            time_[i,k] = time.time() - start_time

            if test_mode == 'mid':
                test_data = torch.from_numpy(all_data[1::2].reshape(-1,1)).type(torch.FloatTensor)
            elif test_mode == 'rest':
                test_data = torch.from_numpy(all_data.reshape(-1,1)).type(torch.FloatTensor)
            elif test_mode == 'all':
                test_data = torch.from_numpy(all_data.reshape(-1,1)).type(torch.FloatTensor)
            else:
                print('Wrong test_mode!')
                return -1

            
            test_data = encoding_func(test_data).to(device)
            
            trn_psnr_[i,k] = psnr_func((train_data@W@train_data.T).transpose(0,2),train_label).detach().cpu().numpy()
            tst_psnr_[i,k] = psnr_func((test_data@W@test_data.T).transpose(0,2).reshape(-1,3)[test_mask],test_label[test_mask]).detach().cpu().numpy()
            
            if logger == None:
                print("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
            else:
                logger.info("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
        rec_[i] = (test_data@W@test_data.T).transpose(0,2).detach().cpu()
    return time_,trn_psnr_,tst_psnr_,rec_




def train_random_simple_2D(signals,encoding_func,ratio=0.25,mask=None,epochs=2000,lr=1e-2,criterion=nn.MSELoss(),device="cpu",
                depth=4,width=256,bias=True,use_sigmoid=False,N_repeat=1,test_mode='rest',logger=None):

    '''
    Trainer of simple encoding for 2D image reconstruction with randomly sampled points.

    The input args are four parts:

    signals: a torch tensor shape in N_image x sample_N x sample_N x 3
    encoding_func: encoding function
    ratio: if mask is None, randomly sample points for training with this ratio of number of all points
    mask: a boolean matrix shape in N_image x N_repeat x (sample_N x sample_N)
    
    epochs,lr,criterion are optimization settings

    depth,width,bias,use_sigmoid are network cofiguration

    N_repeat: number of repeat times for each example
    test_mode: default as 'rest' to use all points except training ones for test. 'mid' will use a shift grid of training points. 'all' will use all ponts including training ones.
    logger: if is None, loggers will print out.

    Returns:

    running time: N_test x N_repeat
    train psnr: N_test x N_repeat
    test psnr: N_test x N_repeat
    reconstuction results: same size as input signals
    '''

    
    sample_N =signals.shape[1]
    N_test = signals.shape[0]

    trn_psnr_ = np.zeros((N_test,N_repeat))
    tst_psnr_ = np.zeros((N_test,N_repeat))
    time_ = np.zeros((N_test,N_repeat))
    rec_ = np.zeros((N_test,sample_N,sample_N,3))

    # build the meshgrid for coordinates
    x1 = np.linspace(0, 1, sample_N+1)[:-1]
    all_data = np.stack(np.meshgrid(x1,x1), axis=-1)
    


    for i in range(N_test):
        
        for k in range(N_repeat):

            if mask is None:
                idx = torch.randperm(sample_N**2)[:int(ratio*sample_N**2)]
                mask_np_N2 = np.zeros((sample_N**2))
                mask_np_N2[idx] = 1
                mask_np_N2 = mask_np_N2==1
            else:
                mask_np_N2 = mask[i,k]

            # Prepare the targets
            all_target = signals[i].squeeze()
            train_label = all_target.reshape(-1,3)[mask_np_N2].type(torch.FloatTensor)
            
            start_time = time.time()
            train_data = encoding_func(torch.from_numpy(all_data.reshape(-1,2)[mask_np_N2]).type(torch.FloatTensor))
            train_data, train_label = train_data.to(device),train_label.to(device)


            # Initialize classification model to learn
            model = MLP(input_dim=encoding_func.dim,output_dim=3,depth=depth,width=width,bias=bias,use_sigmoid=use_sigmoid).to(device)
            # Set the optimization
            optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999),weight_decay=1e-8)
            
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                out = model(train_data)
                loss = criterion(out, train_label)

                loss.backward()
                optimizer.step()


            time_[i,k] = time.time() - start_time


            if test_mode == 'rest':
                test_label = all_target.reshape(-1,3)[~mask_np_N2].type(torch.FloatTensor)
                test_data = encoding_func(torch.from_numpy(all_data.reshape(-1,2)[~mask_np_N2]).type(torch.FloatTensor))
            elif test_mode == 'all':
                test_label = all_target.reshape(-1,3).type(torch.FloatTensor)
                test_data = encoding_func(torch.from_numpy(all_data.reshape(-1,2)).type(torch.FloatTensor))
            else:
                print('Wrong test_mode!')
                return -1
            
            test_data, test_label = test_data.to(device),test_label.to(device)

            trn_psnr_[i,k] = psnr_func(model(train_data),train_label)
            tst_psnr_[i,k] = psnr_func(model(test_data),test_label)
                
            if logger == None:
                print("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
            else:
                logger.info("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
        rec_[i] = model(encoding_func(torch.from_numpy(all_data.reshape(-1,2)).type(torch.FloatTensor)).to(device)).reshape(sample_N,sample_N,3).detach().cpu()
            
    return time_, trn_psnr_,tst_psnr_,rec_




def train_index_blend_kron_2D(signals,outter_blending,inner_encoding,ratio=0.25,mask=None,sm=0.0,epochs=2000,lr=1e-1,criterion=torch.nn.MSELoss(),device="cpu",
                depth=0,width=256,width0=256,bias=True,use_sigmoid=False,N_repeat=1,test_mode='rest',logger=None):

    '''
    Trainer of complex encoding for 2D image reconstruction with randomly sampled points.

    The input args are four parts:

    signals: a pytorch tensor shape in N_image x sample_N x sample_N x 3
    inner encoding: encoding function function of the virtual grid points, so the encoded results is static
    outter_blending: blending funcion for interpolation index and weights of random samples
    ratio: if mask is None, randomly sample points for training with this ratio of number of all points
    mask: a boolean matrix shape in N_image x N_repeat x (sample_N x sample_N)
    sm: how much we punish on total variation loss
    
    epochs,lr,criterion are optimization settings

    depth,width,width0,bias,use_sigmoid are network cofiguration

    N_repeat: number of repeat times for each example
    test_mode: default as 'rest' to use all points except training ones for test. 'mid' will use a shift grid of training points. 'all' will use all ponts including training ones.
    logger: if is None, loggers will print out.

    Returns:

    running time: N_test x N_repeat
    train psnr: N_test x N_repeat
    test psnr: N_test x N_repeat
    reconstuction results: same size as input signals
    '''

    
    sample_N =signals.shape[1]
    N_test = signals.shape[0]

    trn_psnr_ = np.zeros((N_test,N_repeat))
    tst_psnr_ = np.zeros((N_test,N_repeat))
    time_ = np.zeros((N_test,N_repeat))
    rec_ = np.zeros((N_test,sample_N,sample_N,3))
    
    # the length of all_data is same as 1D signal length
    x1 = np.linspace(0, 1, sample_N+1)[:-1]
    all_data = np.stack(np.meshgrid(x1,x1), axis=-1)
    all_grid = torch.linspace(0, 1, outter_blending.dim+1)[:-1]
    encoded_grid = inner_encoding(all_grid.reshape(-1,1))

    filter=torch.tensor([[[1.0,-1.0]]]).to(device)


    for i in range(N_test):
        
        for k in range(N_repeat):

            if mask is None:
                idx = torch.randperm(sample_N**2)[:int(ratio*sample_N**2)]
                mask_np_N2 = np.zeros((sample_N**2))
                mask_np_N2[idx] = 1
                mask_np_N2 = mask_np_N2==1
            else:
                mask_np_N2 = mask[i,k]


            # Prepare the targets
            all_target = signals[i].squeeze()
            train_label = all_target.reshape(-1,3)[mask_np_N2].type(torch.FloatTensor)
            
            start_time = time.time()
            train_idx, train_data = outter_blending(torch.from_numpy(all_data.reshape(-1,2)[mask_np_N2]).type(torch.FloatTensor))
            train_idx,train_data, train_label = train_idx.to(device),train_data.to(device),train_label.to(device)
            
            encoded_grid = encoded_grid.to(device)

            # Initialize classification model to learn
            model = Indexing_Blend_Kron_MLP(input_dim=inner_encoding.dim,output_dim=3,depth=depth,width0=width0,width=width,bias=bias,use_sigmoid=use_sigmoid).to(device)
            # Set the optimization
            optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999),weight_decay=1e-8)
                
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                out = model([train_idx,train_data],encoded_grid)
                loss = criterion(out, train_label)
                if sm>0: 
                    for oo in range(model.first.weight.shape[0]):
                        loss += sm*smooth(model.first.weight[oo],filter)

                loss.backward()
                optimizer.step()

                model.eval()
            
            time_[i,k] = time.time() - start_time


            model.eval()
            if test_mode == 'rest':
                test_label = all_target.reshape(-1,3)[~mask_np_N2].type(torch.FloatTensor)
                test_idx,test_data = outter_blending(torch.from_numpy(all_data.reshape(-1,2)[~mask_np_N2]).type(torch.FloatTensor))
            elif test_mode == 'all':
                test_label = all_target.reshape(-1,3).type(torch.FloatTensor)
                test_idx,test_data = outter_blending(torch.from_numpy(all_data.reshape(-1,2)).type(torch.FloatTensor))
            else:
                print('Wrong test_mode!')
                return -1

            test_idx,test_data, test_label = test_idx.to(device),test_data.to(device),test_label.to(device)
            with torch.no_grad():
                trn_psnr_[i,k] = psnr_func(model([train_idx,train_data],encoded_grid),train_label)
                
                tst_psnr_[i,k] = psnr_func(model([test_idx,test_data],encoded_grid),test_label)
                
            if logger == None:
                print("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
            else:
                logger.info("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))

        rec_[i] = model([outter_blending(torch.from_numpy(all_data.reshape(-1,2)).type(torch.FloatTensor))[0].to(device),
                        outter_blending(torch.from_numpy(all_data.reshape(-1,2)).type(torch.FloatTensor))[1].to(device)],encoded_grid).reshape(sample_N,sample_N,3).detach().cpu()
    return time_,trn_psnr_,tst_psnr_,rec_




def train_simple_3D(signals,encoding_func,epochs=500,lr=5e-3,criterion=nn.MSELoss(),device="cpu",
                depth=5,width=512,bias=True,use_sigmoid=False,N_repeat=1,test_mode='rest',logger=None):

    '''
    Trainer of simple encoding for 3D video reconstruction with separable coordinates.

    The input args are four parts:

    signals: a pytorch tensor shape in N_video x sample_N x sample_N x sample_N x 3
    encoding_func: encoding function
    
    epochs,lr,criterion are optimization settings

    depth,width,bias,use_sigmoid are network cofiguration

    N_repeat: number of repeat times for each example
    test_mode: default as 'rest' to use all points except training ones for test. 'mid' will use a shift grid of training points. 'all' will use all ponts including training ones.
    logger: if is None, loggers will print out.

    Returns:

    running time: N_test x N_repeat
    train psnr: N_test x N_repeat
    test psnr: N_test x N_repeat
    reconstuction results: same size as input signals
    '''

    
    sample_N =signals.shape[1]
    N_test = signals.shape[0]

    trn_psnr_ = np.zeros((N_test,N_repeat))
    tst_psnr_ = np.zeros((N_test,N_repeat))
    time_ = np.zeros((N_test,N_repeat))
    rec_ = np.zeros((N_test,sample_N,sample_N,sample_N,3))

    # build the meshgrid for 3D coordinates
    x1 = torch.linspace(0, 1, sample_N+1)[:-1]
    all_data = torch.stack(torch.meshgrid(x1,x1,x1), axis=-1)
    

    for i in range(N_test):
        # Prepare the targets
        all_target = signals[i].squeeze()
        train_label = all_target[::2,::2,::2].reshape(-1,3).type(torch.FloatTensor)
        if test_mode == 'mid':
            test_label = all_target[1::2,1::2,1::2].reshape(-1,3).type(torch.FloatTensor)
        elif test_mode == 'rest':
            test_label1 = all_target[::2,1::2,::2].reshape(-1,3).type(torch.FloatTensor)
            test_label2 = all_target[::2,1::2,1::2].reshape(-1,3).type(torch.FloatTensor)
            test_label3 = all_target[::2,::2,1::2].reshape(-1,3).type(torch.FloatTensor)
            test_label4 = all_target[1::2,::2,::2].reshape(-1,3).type(torch.FloatTensor)
            test_label5 = all_target[1::2,1::2,::2].reshape(-1,3).type(torch.FloatTensor)
            test_label6 = all_target[1::2,1::2,1::2].reshape(-1,3).type(torch.FloatTensor)
            test_label7 = all_target[1::2,::2,1::2].reshape(-1,3).type(torch.FloatTensor)
            test_label = torch.cat([test_label1,test_label2,test_label3,test_label4,test_label5,test_label6,test_label7],0)
        elif test_mode == 'all':
            test_label = all_target.reshape(-1,3).type(torch.FloatTensor)
        
        for k in range(N_repeat):
            start_time = time.time()
            train_data = all_data[::2,::2,::2].reshape(-1,3)
            
            
            if test_mode == 'mid':
                test_data = all_data[1::2,1::2,1::2].reshape(-1,3)
            elif test_mode == 'rest':
                test_data1 = all_data[::2,1::2,::2].reshape(-1,3)
                test_data2 = all_data[::2,1::2,1::2].reshape(-1,3)
                test_data3 = all_data[::2,::2,1::2].reshape(-1,3)
                test_data4 = all_data[1::2,::2,::2].reshape(-1,3)
                test_data5 = all_data[1::2,1::2,::2].reshape(-1,3)
                test_data6 = all_data[1::2,1::2,1::2].reshape(-1,3)
                test_data7 = all_data[1::2,::2,1::2].reshape(-1,3)
                test_data = torch.cat([test_data1,test_data2,test_data3,test_data4,test_data5,test_data6,test_data7],0)
            elif test_mode == 'all':
                test_data = all_data.reshape(-1,3)
                
            else:
                print('Wrong test_mode!')
                return -1

            train_data, train_label = encoding_func(train_data).to(device),train_label.to(device)
            test_data, test_label = encoding_func(test_data).to(device),test_label.to(device)

            # Initialize classification model to learn
            model = MLP(input_dim=encoding_func.dim,output_dim=3,depth=depth,width=width,bias=bias,use_sigmoid=use_sigmoid).to(device)
            # Set the optimization
            optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999),weight_decay=1e-8)
            
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                out = model(train_data)
                loss = criterion(out, train_label)

                loss.backward()
                optimizer.step()
            
            
            time_[i,k] = time.time() - start_time
            model.eval()
            with torch.no_grad():
                trn_psnr_[i,k] = psnr_func(model(train_data),train_label)
                tst_psnr_[i,k] = psnr_func(model(test_data),test_label)
            if logger == None:
                print("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
            else:
                logger.info("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))

        with torch.no_grad():
            for ff in range(128):
                rec_[i,ff] = model(encoding_func((all_data[ff,:,:].reshape(-1,3)).type(torch.FloatTensor)).to(device)).reshape(sample_N,sample_N,3).detach().cpu()

    return time_, trn_psnr_,tst_psnr_,rec_




def train_kron_3D(signals,encoding_func,epochs=500,lr=1e-1,criterion=nn.MSELoss(),device="cpu",
                depth=0,width=256,width0=256,bias=True,use_sigmoid=False,N_repeat=1,test_mode='rest',logger=None):

    '''
    Trainer of complex encoding for 3D video reconstruction with separable coordinates.

    The input args are four parts:

    signals: a pytorch tensor shape in N_video x sample_N x sample_N x sample_N x 3
    encoding_func: encoding function
    
    epochs,lr,criterion are optimization settings

    depth,width,width0,bias,use_sigmoid are network cofiguration

    N_repeat: number of repeat times for each example
    test_mode: default as 'rest' to use all points except training ones for test. 'mid' will use a shift grid of training points. 'all' will use all ponts including training ones.
    logger: if is None, loggers will print out.

    Returns:

    running time: N_test x N_repeat
    train psnr: N_test x N_repeat
    test psnr: N_test x N_repeat
    reconstuction results: same size as input signals
    '''

    
    sample_N =signals.shape[1]
    N_test = signals.shape[0]

    trn_psnr_ = np.zeros((N_test,N_repeat))
    tst_psnr_ = np.zeros((N_test,N_repeat))
    time_ = np.zeros((N_test,N_repeat))
    rec_ = np.zeros((N_test,sample_N,sample_N,sample_N,3))

    # Here we only use 1D grids
    all_data = np.linspace(0, 1, sample_N+1)[:-1]


    for i in range(N_test):
        # Prepare the targets
        all_target = signals[i].squeeze()
        train_label = all_target[::2,::2,::2].reshape(-1,3).type(torch.FloatTensor).to(device)
        if test_mode == 'mid':
            test_label = all_target[1::2,1::2,1::2].reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[True]],device=device).repeat(int(sample_N/2),int(sample_N/2),int(sample_N/2)).reshape(-1)
        elif test_mode == 'rest':
            test_label = all_target.reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[[False,True],[True,True]],[[True,True],[True,True]]],device=device).repeat(int(sample_N/2),int(sample_N/2),int(sample_N/2)).reshape(-1)
        elif test_mode == 'all':
            test_label = all_target.reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[True]],device=device).repeat(sample_N,sample_N,sample_N).reshape(-1)
            
        
        for k in range(N_repeat):
            start_time = time.time()
            # train data are sampled by , test data is either in middle or all data, both are encoded by encoding func
            train_data = torch.from_numpy(all_data[::2].reshape(-1,1)).type(torch.FloatTensor)
            
            if test_mode == 'mid':
                test_data = torch.from_numpy(all_data[1::2].reshape(-1,1)).type(torch.FloatTensor)
            elif test_mode == 'rest':
                test_data = torch.from_numpy(all_data.reshape(-1,1)).type(torch.FloatTensor)
            elif test_mode == 'all':
                test_data = torch.from_numpy(all_data.reshape(-1,1)).type(torch.FloatTensor)
            else:
                print('Wrong test_mode!')
                return -1


            # Initialize classification model to learn
            model = Kron3_MLP(input_dim=encoding_func.dim,output_dim=3,depth=depth,width0=width0,width=width,bias=bias,use_sigmoid=use_sigmoid).to(device)
            # Set the optimization
            optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999),weight_decay=1e-8)

            train_data = encoding_func(train_data).to(device)
            test_data = encoding_func(test_data).to(device)
                
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                out = model(train_data)
                loss = criterion(out, train_label)

                loss.backward()
                optimizer.step()


            time_[i,k] = time.time() - start_time
            model.eval()
            with torch.no_grad():
                trn_psnr_[i,k] = psnr_func(model(train_data),train_label)
                
                tst_psnr_[i,k] = psnr_func(model(test_data)[test_mask],test_label[test_mask])
            
            if logger == None:
                print("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
            else:
                logger.info("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
        with torch.no_grad():
            rec_[i] = model(encoding_func(torch.from_numpy(all_data.reshape(-1,1)).type(torch.FloatTensor)).to(device)).reshape(sample_N,sample_N,sample_N,3).detach().cpu()    
    return time_,trn_psnr_,tst_psnr_,rec_




def train_closed_form_3D(signals,encoding_func,device="cpu",test_mode='rest',N_repeat=1,logger=None):

    '''
    Closed form solution of complex encoding for 3D video reconstruction with separable coordinates. No training.

    The input args are two parts:

    signals: a pytorch tensor shape in N_video x sample_N x sample_N x sample_N x 3
    encoding_func: encoding function

    N_repeat: number of repeat times for each example
    test_mode: default as 'rest' to use all points except training ones for test. 'mid' will use a shift grid of training points. 'all' will use all ponts including training ones.
    logger: if is None, loggers will print out.

    Returns:

    running time: N_test x N_repeat
    train psnr: N_test x N_repeat
    test psnr: N_test x N_repeat
    reconstuction results: same size as input signals
    '''

    
    sample_N =signals.shape[1]
    N_test = signals.shape[0]

    trn_psnr_ = np.zeros((N_test,N_repeat))
    tst_psnr_ = np.zeros((N_test,N_repeat))
    time_ = np.zeros((N_test,N_repeat))
    rec_ = np.zeros((N_test,sample_N,sample_N,sample_N,3))

    # Here we only use 1D grids
    all_data = np.linspace(0, 1, sample_N+1)[:-1]


    for i in range(N_test):
        # Prepare the targets
        all_target = signals[i].squeeze()
        train_label = all_target[::2,::2,::2].type(torch.FloatTensor).to(device)
        if test_mode == 'mid':
            test_label = all_target[1::2,1::2,1::2].reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[True]],device=device).repeat(int(sample_N/2),int(sample_N/2),int(sample_N/2)).reshape(-1)
        elif test_mode == 'rest':
            test_label = all_target.reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[[False,True],[True,True]],[[True,True],[True,True]]],device=device).repeat(int(sample_N/2),int(sample_N/2),int(sample_N/2)).reshape(-1)
        elif test_mode == 'all':
            test_label = all_target.reshape(-1,3).type(torch.FloatTensor).to(device)
            test_mask = torch.tensor([[True]],device=device).repeat(sample_N,sample_N,sample_N).reshape(-1)
          
        
        for k in range(N_repeat):
            start_time = time.time()
            # train data are sampled by , test data is either in middle or all data, both are encoded by encoding func
            train_data = torch.from_numpy(all_data[::2].reshape(-1,1)).type(torch.FloatTensor)
            
            if test_mode == 'mid':
                test_data = torch.from_numpy(all_data[1::2].reshape(-1,1)).type(torch.FloatTensor)
            elif test_mode == 'rest':
                test_data = torch.from_numpy(all_data.reshape(-1,1)).type(torch.FloatTensor)
            elif test_mode == 'all':
                test_data = torch.from_numpy(all_data.reshape(-1,1)).type(torch.FloatTensor)
            else:
                print('Wrong test_mode!')
                return -1


            train_data = encoding_func(train_data).to(device)
            
            test_data = encoding_func(test_data).to(device)
      
            #ix = torch.linalg.pinv(train_data)
            ix = torch.linalg.lstsq(train_data, torch.eye(train_data.shape[0]).to(device)).solution
            W= (ix@(train_label.transpose(0,3))@ix.T).transpose(1,3)@ix.T
            

            time_[i,k] = time.time() - start_time
            trn_psnr_[i,k] = psnr_func((train_data@((W@train_data.T).transpose(1,3))@train_data.T).transpose(0,3),train_label).detach().cpu().numpy()
            tst_psnr_[i,k] = psnr_func((test_data@((W@test_data.T).transpose(1,3))@test_data.T).transpose(0,3).reshape(-1,3)[test_mask],test_label[test_mask]).detach().cpu().numpy()
            
            if logger == None:
                print("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
            else:
                logger.info("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
        rec_[i] = (test_data@((W@test_data.T).transpose(1,3))@test_data.T).transpose(0,3).detach().cpu()
    return time_,trn_psnr_,tst_psnr_,rec_




def train_random_simple_3D(signals,encoding_func,ratio=0.125,mask=None,epochs=500,lr=5e-3,criterion=nn.MSELoss(),device="cpu",
                depth=5,width=512,bias=True,use_sigmoid=False,N_repeat=1,test_mode='rest',logger=None):

    '''
    Trainer of simple encoding for 3D video reconstruction with randomly sampled points.

    The input args are four parts:

    signals: a torch tensor shape in N_video x sample_N x sample_N x sample_N x 3
    encoding_func: encoding function
    ratio: if mask is None, randomly sample points for training with this ratio of number of all points
    mask: a boolean matrix shape in N_video x N_repeat x (sample_N x sample_N x sample_N)
    
    epochs,lr,criterion are optimization settings

    depth,width,bias,use_sigmoid are network cofiguration

    N_repeat: number of repeat times for each example
    test_mode: default as 'rest' to use all points except training ones for test. 'mid' will use a shift grid of training points. 'all' will use all ponts including training ones.
    logger: if is None, loggers will print out.

    Returns:

    running time: N_test x N_repeat
    train psnr: N_test x N_repeat
    test psnr: N_test x N_repeat
    reconstuction results: same size as input signals
    '''

    
    sample_N =signals.shape[1]
    N_test = signals.shape[0]


    trn_psnr_ = np.zeros((N_test,N_repeat))
    tst_psnr_ = np.zeros((N_test,N_repeat))
    time_ = np.zeros((N_test,N_repeat))
    rec_ = np.zeros((N_test,sample_N,sample_N,sample_N,3))

    # build 3D coordinate meshgrid
    x1 = np.linspace(0, 1, sample_N+1)[:-1]
    all_data = np.stack(np.meshgrid(x1,x1,x1), axis=-1)
    

    for i in range(N_test):
        for k in range(N_repeat):
            if mask is None:
                idx = torch.randperm(sample_N**3)[:int(ratio*sample_N**3)]
                mask_np_N2 = np.zeros((sample_N**3))
                mask_np_N2[idx] = 1
                mask_np_N2 = mask_np_N2==1
            else:
                mask_np_N2 = mask[i,k]

            # Prepare the targets
            all_target = signals[i].squeeze()
            train_label = all_target.reshape(-1,3)[mask_np_N2].type(torch.FloatTensor)
            if test_mode == 'rest':
                test_label = all_target.reshape(-1,3)[~mask_np_N2].type(torch.FloatTensor)
            elif test_mode == 'all':
                test_label = all_target.reshape(-1,3).type(torch.FloatTensor)
        
        
            start_time = time.time()
            train_data = encoding_func(torch.from_numpy(all_data.reshape(-1,3)[mask_np_N2]).type(torch.FloatTensor))
            
            
            if test_mode == 'rest':
                test_data = encoding_func(torch.from_numpy(all_data.reshape(-1,3)[~mask_np_N2]).type(torch.FloatTensor))
            elif test_mode == 'all':
                test_data = encoding_func(torch.from_numpy(all_data.reshape(-1,3)).type(torch.FloatTensor))
                
            else:
                print('Wrong test_mode!')
                return -1

            train_data, train_label = train_data.to(device),train_label.to(device)
            test_data, test_label = test_data.to(device),test_label.to(device)


            # Initialize classification model to learn
            model = MLP(input_dim=encoding_func.dim,output_dim=3,depth=depth,width=width,bias=bias,use_sigmoid=use_sigmoid).to(device)
            # Set the optimization
            optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999),weight_decay=1e-8)
            
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                out = model(train_data)
                loss = criterion(out, train_label)

                loss.backward()
                optimizer.step()
            time_[i,k] = time.time() - start_time
            model.eval()
            with torch.no_grad():
                trn_psnr_[i,k] = psnr_func(model(train_data),train_label)
                tst_psnr_[i,k] = psnr_func(model(test_data),test_label)
            
                
            if logger == None:
                print("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
            else:
                logger.info("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
        with torch.no_grad():
            for ff in range(128):
                rec_[i,ff] = model(encoding_func(torch.from_numpy(all_data[ff,:,:].reshape(-1,3)).type(torch.FloatTensor)).to(device)).reshape(sample_N,sample_N,3).detach().cpu()   
    return time_, trn_psnr_,tst_psnr_,rec_




def train_index_blend_kron_3D(signals,outter_blending,inner_encoding,ratio=0.125,mask=None,sm=0.0,epochs=500,lr=1e-1,criterion=torch.nn.MSELoss(),device="cpu",
                depth=0,width=256,width0=256,bias=True,use_sigmoid=False,N_repeat=1,test_mode='rest',logger=None):

    '''
    Trainer of complex encoding for 3D video reconstruction with randomly sampled points.

    The input args are four parts:

    signals: a torch tensor shape in N_video x sample_N x sample_N x sample_N x 3
    encoding_func: encoding function
    ratio: if mask is None, randomly sample points for training with this ratio of number of all points
    mask: a boolean matrix shape in N_video x N_repeat x (sample_N x sample_N x sample_N)
    sm: how much we punish on total variation loss
    
    epochs,lr,criterion are optimization settings

    depth,width,width0,bias,use_sigmoid are network cofiguration

    N_repeat: number of repeat times for each example
    test_mode: default as 'rest' to use all points except training ones for test. 'mid' will use a shift grid of training points. 'all' will use all ponts including training ones.
    logger: if is None, loggers will print out.

    Returns:

    running time: N_test x N_repeat
    train psnr: N_test x N_repeat
    test psnr: N_test x N_repeat
    reconstuction results: same size as input signals
    '''

    
    sample_N =signals.shape[1]
    N_test = signals.shape[0]


    trn_psnr_ = np.zeros((N_test,N_repeat))
    tst_psnr_ = np.zeros((N_test,N_repeat))
    time_ = np.zeros((N_test,N_repeat))
    rec_ = np.zeros((N_test,sample_N,sample_N,sample_N,3))
    
    # the length of all_data is same as 1D signal length
    x1 = np.linspace(0, 1, sample_N+1)[:-1]
    all_data = np.stack(np.meshgrid(x1,x1,x1), axis=-1)
    all_grid = torch.linspace(0, 1, outter_blending.dim+1)[:-1]
    encoded_grid = inner_encoding(all_grid.reshape(-1,1))

    filter=torch.tensor([[[1.0,-1.0]]]).to(device)


    for i in range(N_test):
        
        for k in range(N_repeat):

            if mask is None:
                idx = torch.randperm(sample_N**3)[:int(ratio*sample_N**3)]
                mask_np_N2 = np.zeros((sample_N**3))
                mask_np_N2[idx] = 1
                mask_np_N2 = mask_np_N2==1
            else:
                mask_np_N2 = mask[i,k]


            # Prepare the targets
            all_target = signals[i].squeeze()
            train_label = all_target.reshape(-1,3)[mask_np_N2].type(torch.FloatTensor)
            if test_mode == 'rest':
                test_label = all_target.reshape(-1,3)[~mask_np_N2].type(torch.FloatTensor)
            elif test_mode == 'all':
                test_label = all_target.reshape(-1,3).type(torch.FloatTensor)
        
        
            start_time = time.time()
            train_idx,train_data = outter_blending(torch.from_numpy(all_data.reshape(-1,3)[mask_np_N2]).type(torch.FloatTensor))
            
            
            if test_mode == 'rest':
                test_idx, test_data = outter_blending(torch.from_numpy(all_data.reshape(-1,3)[~mask_np_N2]).type(torch.FloatTensor))
            elif test_mode == 'all':
                test_idx, test_data = outter_blending(torch.from_numpy(all_data.reshape(-1,3)).type(torch.FloatTensor))
                
            else:
                print('Wrong test_mode!')
                return -1

            train_idx, train_data, train_label = train_idx.to(device), train_data.to(device),train_label.to(device)
            test_idx, test_data, test_label = test_idx.to(device), test_data.to(device),test_label.to(device)
            encoded_grid = encoded_grid.to(device)

            # Initialize classification model to learn
            model = Indexing_Blend_Kron3_MLP(input_dim=inner_encoding.dim,output_dim=3,depth=depth,width0=width0,width=width,bias=bias,use_sigmoid=use_sigmoid).to(device)
            # Set the optimization
            optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999),weight_decay=1e-8)
                
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                out = model([train_idx,train_data],encoded_grid)
                loss = criterion(out, train_label)
                if sm>0: 
                    for oo in range(model.first.weight.shape[0]):
                        loss += sm*smooth_3D(model.first.weight[oo],filter)

                loss.backward()
                optimizer.step()

            time_[i,k] = time.time() - start_time
            model.eval()
            with torch.no_grad():
                trn_psnr_[i,k] = psnr_func(model([train_idx,train_data],encoded_grid),train_label)
                tst_psnr_[i,k] = psnr_func(model([test_idx,test_data],encoded_grid),test_label)
                
            if logger == None:
                print("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
            else:
                logger.info("==>>> E: %g, T: %g, train psnr: %g--- , test psnr: %g--- , time: %g seconds ---" 
                    % (i,k, np.mean(trn_psnr_[i,k]),np.mean(tst_psnr_[i,k]),time.time() - start_time))
        with torch.no_grad():
            rec_[i] = model([outter_blending(torch.from_numpy(all_data.reshape(-1,3)).type(torch.FloatTensor))[0].to(device),
                        outter_blending(torch.from_numpy(all_data.reshape(-1,3)).type(torch.FloatTensor))[1].to(device)],encoded_grid).reshape(sample_N,sample_N,sample_N,3).detach().cpu()
    return time_,trn_psnr_,tst_psnr_,rec_


