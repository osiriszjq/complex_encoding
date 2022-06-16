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
    
    signals = torch.from_numpy(buf['test'])

    logger.info('################ Simple Encoding ################')

    ez = 192
    rff_params = [3]
    linearf_params = [3]
    logf_params = [3]
    gaussian_params = [0.015]
    linear_params = [3/64]

    encoding_methods = ['RFF']*len(rff_params)+['LinF']*len(linearf_params)+['LogF']*len(logf_params)+['Gau']*len(gaussian_params)+['Tri']*len(linear_params)
    params = rff_params+linearf_params+logf_params+gaussian_params+linear_params

    for depth in [5,1,0]:
        for lr in [5e-3]:
            logger.info('######## Network Depth = {}, Learning Rate = {} ########'.format(depth,lr))
            for em,param in zip(encoding_methods,params):

                ef = encoding_func_3D(em,[param,ez])
                time_,trn_psnr_,tst_psnr_,rec_ = train_simple_3D(signals,ef,N_repeat=args.N_repeat,lr=lr,epochs=500,depth=depth,device=device,logger=None)
                file_name = 'D{}{}'.format(depth,em)
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
    linearf_params = [3]
    logf_params = [3]
    gaussian_params = [0.015]
    linear_params = [2/64]

    encoding_methods = ['RFF']*len(rff_params)+['LinF']*len(linearf_params)+['LogF']*len(logf_params)+['Gau']*len(gaussian_params)+['Tri']*len(linear_params)
    params = rff_params+linearf_params+logf_params+gaussian_params+linear_params

    for depth in [0,1]:
        for lr in [1e-1]:
            logger.info('######## Network Depth = {}, Learning Rate = {} ########'.format(depth,lr))
            for em,param in zip(encoding_methods,params):

                ef = encoding_func_1D(em,[param,ez])
                time_,trn_psnr_,tst_psnr_,rec_ = train_kron_3D(signals,ef,N_repeat=args.N_repeat,lr=lr,epochs=500,depth=depth,device=device,logger=None)
                file_name = 'KD{}{}'.format(depth,em)
                if args.save_flag:
                    # np.save(args.save_path+file_name+'_rec.npy',rec_)
                    # np.save(args.save_path+file_name+'_time.npy',time_)
                    # np.save(args.save_path+file_name+'_trn.npy',trn_psnr_)
                    # np.save(args.save_path+file_name+'_tst.npy',tst_psnr_)
                    for i in range(signals.shape[0]):
                        imageio.mimwrite(args.save_path+'V{}'.format(i)+file_name+ 'R{:.2f}.mp4'.format(tst_psnr_[i,-1]), to8b(rec_[i]), fps=10, quality=8)

                logger.info('embedding method:{}, param:{}, psnr:{}, std:{}, time:{}.'.format(em,param,np.mean(tst_psnr_[:,:]),np.std(tst_psnr_[:,:]),np.mean(time_[:,:])))
    

    logger.info('################ Closed Form Complex Encoding ################')

    ez = 64
    rff_params = [6]
    linearf_params = [4]
    logf_params = [3]
    gaussian_params = [0.009]
    linear_params = [3/64]

    for em,param in zip(encoding_methods,params):

        if em == 'Gau' or em == 'Tri': em=em
        else: device = 'cpu'

        ef = encoding_func_1D(em,[param,ez])
        time_,trn_psnr_,tst_psnr_,rec_ = train_closed_form_3D(signals,ef,N_repeat=args.N_repeat,device=device,logger=None)
        file_name = 'CKD{}{}'.format(depth,em)
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
    parser.add_argument("--N_repeat", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="3D_separable_points/")
    parser.add_argument("--logger", type=str, default="log.log")
    parser.add_argument("--save_flag", type=int, default=0, choices=[0, 1])


    args = parser.parse_args()
    main(args)