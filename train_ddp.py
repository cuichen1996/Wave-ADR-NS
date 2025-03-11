import json
import logging
import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from data.load_data import CreateDataLoader
from model.model import WaveADR
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.misc import *


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(self, model, gpu_id, config) -> None:
        self.run_dir = config['run_dir']
        self.logger = logging.getLogger(__name__)
        
        self.max_epochs = config['epoch'] 
        self.gpu_id = gpu_id
        self.train_loader = CreateDataLoader(config)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['step_size'], gamma=config['gamma'])

        if config['restart']:
            checkpoint = config['restart_dir'] + '/model_{}.pth'.format(config['restart_epoch'])
            model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint).items()})
            self.logger.info('Loaded pre-trained model: {}'.format(checkpoint))  
            
        self.model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        self.checkpoints_folder = config['checkpoints_folder']
        self.restart_dir = config['restart_dir']
        self.restart_epoch = config['restart_epoch']
        self.save_every = config['ckpt_freq'] 
        self.clip = config['grad_clip']

    def _run_batch(self, f, kappa, u):
        self.optimizer.zero_grad()
        loss = self.model(f, kappa, u)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch, count):
        train_loss = 0.0
        epoch_start_time = time.time()
        for f, kappa, u in self.train_loader[count]:
            f = f.to(self.gpu_id)
            kappa = kappa.to(self.gpu_id)
            u = u.to(self.gpu_id)
            loss = self._run_batch(f, kappa, u)
            train_loss += (loss*f.shape[0])
        train_loss /= len(self.train_loader[count].dataset)
        self.logger.info("epoch: [{:d}/{:d}] {:.2f} sec(s) Train Loss: {:.9f} ".format(epoch, self.max_epochs, time.time()-epoch_start_time, train_loss))

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict() #?
        PATH = self.checkpoints_folder + "/model_"+str(epoch)+".pth"
        torch.save(ckp, PATH)

    def train(self,):
        count = 0
        self.logger = self.config_logger()
        for epoch in range(1, self.max_epochs+1):
            self._run_epoch(epoch, count)
            self.scheduler.step()
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            if epoch > 0 and epoch % 20 == 0:
                count += 1
                if count > 1:
                    count = 0
                print(epoch)

    def config_logger(self):
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler = logging.FileHandler(self.run_dir + "/train.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        return self.logger
    

def main(rank, world_size, config):
    ddp_setup(rank, world_size)
    
    # * model
    model = WaveADR(config)
    total_params = count_parameters(model)
    print("Total parameters: ", total_params)
    
    trainer = Trainer(model, rank, config)
    trainer.train()
    destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print("Using", world_size, "GPUs.")
    setup_seed(1234)
    
    # *=======================参数设置=================================
    config = {}

    # data  
    config['batch_size'] = 32

    # model
    config["M"]              = 1
    config["NO_Type"]        = "FNO" 
    config["modes"]          = [12,12,12,12]
    config["depths"]         = [3,3,9,3]
    config["dims"]           = [36,36,36,36]
    config["drop_path_rate"] = 0.3
    config["drop"]           = 0.
    config["padding"]        = 9
    config["act"]            = "gelu"
    config["xavier_init"]    = 1e-2

    config["max_iter_num"]    = 100
    config["error_threshold"] = 1e-6
    
    # dir
    config['run_dir'] = "expriments/WaveADR_{}".format(config["NO_Type"])
    config['checkpoints_folder'] = config['run_dir'] + '/checkpoints'
    config['prediction_folder'] = config['run_dir'] + '/prediction'
    config['restart_dir'] = config['checkpoints_folder'] 

    # train
    config['grad_clip']     = 1
    config['lr']            = 1e-4
    config['step_size']     = 120
    config['gamma']         = 0.5
    config['epoch']         = 10000
    config['ckpt_freq']     = 10
    config['restart_epoch'] = 90
    
    # 
    config['restart'] = True
    config['restart'] = False

    mkdirs([config['run_dir'], config['checkpoints_folder'], config['prediction_folder']])

    # *=================================================================
    import time
    logging.info('Start training..............................................')
    tic = time.time()
    mp.spawn(main, args=(world_size, config), nprocs=world_size)
    tic2 = time.time()
    logging.info("Finished training {} epochs using {} seconds".format(config['epoch'], tic2 - tic))
    logging.info("This time training parameter setting: ")
    with open(config['run_dir'] + "/train.log", 'a') as args_file:json.dump(config, args_file, indent=4)