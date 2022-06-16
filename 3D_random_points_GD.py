import torch, os, logging
from argparse import ArgumentParser
import numpy as np

from trainer import *
from utils import *
from MLPs import *


import imageio

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def main(args):

    if os.path.exists(args.save_path):
        print('Path already exists!')
        return 1
    os.mkdir(args.save_path)
    logger = get_logger(args.save_path+args.logger)
    logger.info(args)

    # Set the CUDA flag
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info('device is: {}'.format(device))


    buf = np.load(args.data_path)

    signals = torch.from_numpy(buf['train'])
    
    #mask = np.load(args.mask_path)
    ########### generate random mask ###########
    image_size = signals.shape[1]
    ratio = 0.125
    mask = []
    for i in range(signals.shape[0]):
        mask_tmp = []
        for j in range(args.N_repeat):
            idx = torch.randperm(image_size**3)[:int(ratio*image_size**3)]
            mask_np_N2 = np.zeros((image_size**3))
            mask_np_N2[idx] = 1
            mask_np_N2 = mask_np_N2==1
            mask_tmp.append(mask_np_N2)
        mask.append(np.stack(mask_tmp,0))
    mask = np.stack(mask,0)
    np.save('mask_3d_{}_{}_{}_{}.npy'.format(ratio,signals.shape[0],args.N_repeat,signals.shape[1]),mask)
    ###########################################


    logger.info('################ Simple Encoding ################')

    ez = 192
    rff_params = [2]
    linearf_params = [3]
    logf_params = [3]
    gaussian_params = [0.02]
    linear_params = [3/64]


    encoding_methods = ['RFF']*len(rff_params)+['LinF']*len(linearf_params)+['LogF']*len(logf_params)+['Gau']*len(gaussian_params)+['Tri']*len(linear_params)
    params = rff_params+linearf_params+logf_params+gaussian_params+linear_params


    for depth in [5,1,0]:
        for lr in [5e-3]:
            logger.info('######## Network Depth = {}, Learning Rate = {} ########'.format(depth,lr))
            for em,param in zip(encoding_methods,params):

                ef = encoding_func_3D(em,[param,ez])
                time_,trn_psnr_,tst_psnr_,rec_ = train_random_simple_3D(signals,ef,mask=mask,N_repeat=args.N_repeat,lr=lr,epochs=500,depth=depth,device=device,logger=None)
                file_name = 'RD{}{}'.format(depth,em)
                if args.save_flag:
                    # np.save(args.save_path+file_name+'_rec.npy',rec_)
                    # np.save(args.save_path+file_name+'_time.npy',time_)
                    # np.save(args.save_path+file_name+'_trn.npy',trn_psnr_)
                    # np.save(args.save_path+file_name+'_tst.npy',tst_psnr_)
                    for i in range(signals.shape[0]):
                        imageio.mimwrite(args.save_path+'V{}'.format(i)+file_name+ 'R{:.2f}.mp4'.format(tst_psnr_[i,-1]), to8b(rec_[i]), fps=10, quality=8)

                logger.info('embedding method:{}, param:{}, psnr:{}, std:{}, time:{}.'.format(em,param,np.mean(tst_psnr_[:,:]),np.std(tst_psnr_[:,:]),np.mean(time_[:,:])))

    logger.info('################ Complex Encoding ################')
    
    ez = 64
    rff_params = [6]
    linearf_params = [8]
    logf_params = [8]
    gaussian_params = [0.02]
    linear_params = [3.5/64]

    encoding_methods = ['RFF']*len(rff_params)+['LinF']*len(linearf_params)+['LogF']*len(logf_params)+['Gau']*len(gaussian_params)+['Tri']*len(linear_params)
    params = rff_params+linearf_params+logf_params+gaussian_params+linear_params

    for depth in [0,1]:
        for lr in [1e-1]:
            logger.info('######## Network Depth = {}, Learning Rate = {} ########'.format(depth,lr))
            for em,param in zip(encoding_methods,params):

                ef = encoding_func_1D(em,[param,ez])
                bl = blending_func_3D(ef)
                time_,trn_psnr_,tst_psnr_,rec_ =  train_index_blend_kron_3D(signals,bl,ef,mask=mask,N_repeat=args.N_repeat,lr=lr,epochs=500,depth=depth,device=device,logger=None)
                file_name = 'RKD{}{}'.format(depth,em)
                if args.save_flag:
                    # np.save(args.save_path+file_name+'_rec.npy',rec_)
                    # np.save(args.save_path+file_name+'_time.npy',time_)
                    # np.save(args.save_path+file_name+'_trn.npy',trn_psnr_)
                    # np.save(args.save_path+file_name+'_tst.npy',tst_psnr_)
                    for i in range(signals.shape[0]):
                        imageio.mimwrite(args.save_path+'V{}'.format(i)+file_name+ 'R{:.2f}.mp4'.format(tst_psnr_[i,-1]), to8b(rec_[i]), fps=10, quality=8)

                logger.info('embedding method:{}, param:{}, psnr:{}, std:{}, time:{}.'.format(em,param,np.mean(tst_psnr_[:,:]),np.std(tst_psnr_[:,:]),np.mean(time_[:,:])))
    


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20220222)
    np.random.seed(20220222)


    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str, default="video_16_128.npz")
    parser.add_argument("--mask_path", type=str, default="mask_3d_0.125_5_1_128.npy")
    parser.add_argument("--N_repeat", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="3D_random_points/")
    parser.add_argument("--logger", type=str, default="log.log")
    parser.add_argument("--save_flag", type=int, default=0, choices=[0, 1])


    args = parser.parse_args()
    main(args)