import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from glog import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tools.metric import PSNRMeter
from lib.core.base_trainer.DeAP import DeAP
from lib.core.dataset.dataietr import TrainDataset, ValDataset
from lib.core.base_trainer.utils.loss import PSNR_Loss

class Train(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.ddp = self.cfg.TRAIN.DDP
        self.epochs = self.cfg.TRAIN.epochs
        self.local_rank = 0
        self.PSNRMeter = PSNRMeter()
        if self.ddp:
            torch.distributed.init_process_group(backend='nccl')
            self.local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)

            self.train_dataset = TrainDataset(cfg=cfg)
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.train_dataloader = DataLoader(self.train_dataset, 
                                               batch_size=self.cfg.TRAIN.batch_size, 
                                               num_workers=self.cfg.TRAIN.process_num, 
                                               sampler=self.train_sampler, 
                                               drop_last=True)
            
            self.val_dataset = ValDataset(cfg=cfg)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
            self.val_dataloader = DataLoader(self.val_dataset, 
                                             batch_size=1, 
                                             num_workers=self.cfg.TRAIN.process_num, 
                                             sampler=self.val_sampler)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.train_dataset = TrainDataset(cfg=cfg)

            self.train_dataloader = DataLoader(self.train_dataset, 
                                               batch_size=self.cfg.TRAIN.batch_size, 
                                               num_workers=self.cfg.TRAIN.process_num, 
                                               shuffle=True,
                                               drop_last=True)
            self.val_dataset = ValDataset(cfg=cfg)
            self.val_dataloader = DataLoader(self.val_dataset, 
                                             batch_size=1,
                                             num_workers=self.cfg.TRAIN.process_num, 
                                             shuffle=False)
            
        self.model = DeAP(cfg).to(self.device)
        
        if self.ddp:
            self.model = nn.parallel.DistributedDataParallel(self.model, 
                                                             device_ids=[self.local_rank], 
                                                             output_device=self.local_rank, 
                                                             find_unused_parameters=True)
        else:
            self.model = nn.DataParallel(self.model)
        
        self.load_weight()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                    T_max=self.epochs,
                                                                    eta_min=self.cfg.TRAIN.min_lr)
        self.contrast_loss = nn.CrossEntropyLoss().to(self.device)
        self.psnr_loss = PSNR_Loss().to(self.device)
    
    def load_weight_backbone(self):
        state_dict = torch.load("./pretrained/Stripformer_gopro.pth")
        stripped_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        self.model.module.backbone.load_state_dict(stripped_state_dict, strict=False)

    def load_weight(self):
        state_dict = torch.load("./weights/final_DeAP_pretrain.pth")
        stripped_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        self.netG.module.load_state_dict(stripped_state_dict)

    def train(self, epoch):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        tq = tqdm.tqdm(self.train_dataloader, total=len(self.train_dataloader))
        tq.set_description(f'Epoch {epoch}, lr {lr}')

        loss_list = []
        psnr_list = []
        ssim_list = []
        self.model.train()
        for blur_patch_1, blur_patch_2, sharp_patch_1, sharp_patch_2 in tq:
            blur_patch_1 = blur_patch_1.to(self.device)
            blur_patch_2 = blur_patch_2.to(self.device)
            sharp_patch_1 = sharp_patch_1.to(self.device)
            sharp_patch_2 = sharp_patch_2.to(self.device)
            self.optimizer.zero_grad()
            if epoch < self.cfg.TRAIN.MCFM_epoch:
                _, logits, labels = self.model.module.MCFM(blur_patch_1, blur_patch_2)
                contrast_loss = self.contrast_loss(logits, labels)
                loss = contrast_loss
            else:
                restored, logits, labels = self.model(blur_patch_1, blur_patch_2)
                contrast_loss = self.contrast_loss(logits, labels)
                psnr_loss = self.psnr_loss(restored, sharp_patch_1, sharp_patch_2)
                loss = psnr_loss + 0.1 * contrast_loss
            loss.backward()
            self.optimizer.step()

            if epoch < self.cfg.TRAIN.MCFM_epoch:
                loss_list.append(loss.item())
                tq.set_postfix(loss="{:.4f}".format(np.mean(loss_list)))
            else:
                psnr, ssim = self.PSNRMeter.cal_psnr(restored, sharp_patch_1)         
                loss_list.append(loss.item())
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                tq.set_postfix(loss="{:.4f}".format(np.mean(loss_list)), psnr="{:.4f}".format(np.mean(psnr_list)), ssim="{:.4f}".format(np.mean(ssim_list)))

        if self.ddp:
            loss_tensor = torch.tensor(np.sum(loss_list), device=self.device)
            psnr_tensor = torch.tensor(np.sum(psnr_list), device=self.device)
            ssim_tensor = torch.tensor(np.sum(ssim_list), device=self.device)

            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(psnr_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(ssim_tensor, op=torch.distributed.ReduceOp.SUM)

            world_size = torch.distributed.get_world_size()
            loss_avg = loss_tensor.item() / (len(loss_list) * world_size)
            psnr_avg = psnr_tensor.item() / (len(psnr_list) * world_size)
            ssim_avg = ssim_tensor.item() / (len(ssim_list) * world_size)
            logger.info(f'Epoch {epoch}, Loss: {loss_avg:.4f}, PSNR: {psnr_avg:.4f}, SSIM: {ssim_avg:.4f}')
        tq.close()

    def val(self):
        tq = tqdm.tqdm(self.val_dataloader, total=len(self.val_dataloader))
        tq.set_description('Validation')
        psnr_list = []
        ssim_list = []
        self.model.eval()
        
        for blur, sharp in tq:
            with torch.no_grad():
                blur = blur.to(self.device)
                sharp = sharp.to(self.device)
                restored = self.model(blur, blur)
                psnr, ssim = self.PSNRMeter.cal_psnr(restored, sharp)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                tq.set_postfix(psnr="{:.4f}".format(np.mean(psnr_list)), ssim="{:.4f}".format(np.mean(ssim_list)))
        
        if self.ddp:
            psnr_tensor = torch.tensor(np.sum(psnr_list), device=self.device)
            ssim_tensor = torch.tensor(np.sum(ssim_list), device=self.device)

            torch.distributed.all_reduce(psnr_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(ssim_tensor, op=torch.distributed.ReduceOp.SUM)

            world_size = torch.distributed.get_world_size()
            psnr_avg = psnr_tensor.item() / (len(psnr_list) * world_size)
            ssim_avg = ssim_tensor.item() / (len(ssim_list) * world_size)
            logger.info(f'Validation, PSNR: {psnr_avg:.4f}, SSIM: {ssim_avg:.4f}')

        tq.close()

    def loop(self):
        for epoch in range(self.epochs):
            if self.ddp:
                self.train_sampler.set_epoch(epoch)
            self.train(epoch=epoch)
            if epoch % 200 == 0 or epoch == (self.epochs - 1):
                self.val()
            self.scheduler.step()

            if self.local_rank == 0 and not os.access(self.cfg.MODEL.model_path, os.F_LOCK):
                os.mkdir(self.cfg.MODEL.model_path)
            if self.local_rank == 0 and epoch % 200 == 0:
                torch.save(self.model.state_dict(), self.cfg.MODEL.model_path + '/DeAP_{}.pth'.format(epoch))
            torch.cuda.empty_cache()
