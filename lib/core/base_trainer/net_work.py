import os
import torch
import torch.nn as nn
from glog import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lib.core.dataset.dataietr import TrainDataset, ValDataset
from lib.core.base_trainer.utils.loss import PSNR_Loss
from lib.core.base_trainer.DeAP import DeAP
from tools.metric import PSNRMeter

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
            
        self.model = DeAP().to(self.device)
        
        if self.ddp:
            self.model = nn.parallel.DistributedDataParallel(self.model, 
                                                             device_ids=[self.local_rank], 
                                                             output_device=self.local_rank, 
                                                             find_unused_parameters=True)
        else:
            self.model = nn.DataParallel(self.model)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                    T_max=self.epochs,
                                                                    eta_min=self.cfg.TRAIN.min_lr)
        self.contrast_loss = nn.CrossEntropyLoss().to(self.device)
        self.psnr_loss = PSNR_Loss().to(self.device)

    def train(self, epoch):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        logger.info(lr)
        self.model.train()

        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = len(self.train_dataloader)

        for blur_patch_1, blur_patch_2, sharp_patch_1, sharp_patch_2 in self.train_dataloader:
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
            
            psnr, ssim = self.PSNRMeter.cal_psnr(restored, sharp_patch_1)
            
            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim

        if self.ddp:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_psnr_tensor = torch.tensor(total_psnr, device=self.device)
            total_ssim_tensor = torch.tensor(total_ssim, device=self.device)
            num_batches_tensor = torch.tensor(num_batches, device=self.device)

            torch.distributed.all_reduce(total_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_psnr_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_ssim_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(num_batches_tensor, op=torch.distributed.ReduceOp.SUM)

            total_loss = total_loss_tensor.item() / num_batches_tensor.item()
            total_psnr = total_psnr_tensor.item() / num_batches_tensor.item()
            total_ssim = total_ssim_tensor.item() / num_batches_tensor.item()
        else:
            total_loss /= num_batches
            total_psnr /= num_batches
            total_ssim /= num_batches

        logger.info(f'Epoch {epoch}: Loss {total_loss}, PSNR {total_psnr}, SSIM {total_ssim}')

    def val(self):
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = len(self.val_dataloader)
        
        for blur, sharp in self.val_dataloader:
            with torch.no_grad():
                blur = blur.to(self.device)
                sharp = sharp.to(self.device)
                restored = self.model(blur, blur)
                psnr, ssim = self.PSNRMeter.cal_psnr(restored, sharp)
                
                total_psnr += psnr
                total_ssim += ssim

        if self.ddp:
            total_psnr_tensor = torch.tensor(total_psnr, device=self.device)
            total_ssim_tensor = torch.tensor(total_ssim, device=self.device)
            num_batches_tensor = torch.tensor(num_batches, device=self.device)

            torch.distributed.all_reduce(total_psnr_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_ssim_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(num_batches_tensor, op=torch.distributed.ReduceOp.SUM)

            total_psnr = total_psnr_tensor.item() / num_batches_tensor.item()
            total_ssim = total_ssim_tensor.item() / num_batches_tensor.item()
        else:
            total_psnr /= num_batches
            total_ssim /= num_batches

        logger.info(f'Validation: PSNR {total_psnr}, SSIM {total_ssim}')

    def loop(self):
        for epoch in range(self.epochs):
            if self.ddp:
                self.train_sampler.set_epoch(epoch)
            self.train(epoch=epoch)
            self.val()
            self.scheduler.step()

            if self.local_rank == 0 and not os.access(self.cfg.MODEL.model_path, os.F_LOCK):
                os.mkdir(self.cfg.MODEL.model_path)
            if self.local_rank == 0:
                torch.save(self.model.module.state_dict(), self.save_dir + '{epoch}.pth')
            torch.cuda.empty_cache()
