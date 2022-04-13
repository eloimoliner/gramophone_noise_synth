"""
Train a diffusion model on images.
"""
import os
import hydra
import logging

import torch
import numpy as npp

#from improved_diffusion import dist_util, logger
#from improved_diffusion.image_datasets import load_data
#from improved_diffusion.script_util import (
#    model_and_diffusion_defaults,
#    create_model_and_diffusion,
#    args_to_dict,
#    add_dict_to_argparser,
#)
import dataset_noise_loader

from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from torch.utils.data import DataLoader
import numpy as np

from learner import Learner
from getters import get_sde
from model import UNet

def run(args):

    args = OmegaConf.structured(OmegaConf.to_yaml(args))
    #OmegaConf.set_struct(conf, True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    dirname = os.path.dirname(__file__)
    path_experiment = os.path.join(dirname, str(args.model_dir))

    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)
    args.model_dir=path_experiment


    #logger.log("creating data loader...")
    def worker_init_fn(worker_id):                                                          
        st=np.random.get_state()[2]
        np.random.seed( st+ worker_id)

    dataset_train=dataset_noise_loader.TrainNoiseDataset( args.dset.path_music_train, args.sample_rate,args.audio_len)
    train_loader=DataLoader(dataset_train,num_workers=args.num_workers, batch_size=args.batch_size,  worker_init_fn=worker_init_fn)

    train_set = iter(train_loader)

    model = UNet(args).to(device)

    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    learner = Learner(
        args.model_dir, model, train_set, opt, args
    )

    learner.is_master = True
    #learner.restore_from_checkpoint(params['checkpoint_id'])
    learner.train()


def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    run(args)

@hydra.main(config_path="conf", config_name="conf")
def main(args):
    #try:
        _main(args)
    #except Exception:
        #logger.exception("Some error happened")
    #    os._exit(1)

if __name__ == "__main__":
    main()
