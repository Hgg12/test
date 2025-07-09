import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import time

# 导入必要的模块
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
from lib.train.actors import MobileViTTrackActor
from lib.train.data.loader import create_loader
from lib.models.mobilevit_track.mobilevitv2_track import build_mobilevitv2_track
from lib.utils.box_ops import giou_loss
from lib.config.mobilevitv2_track.config import cfg, update_config_from_file
from lib.train.data.processing import STARKProcessing

class BaseTrainer:
    """ Base trainer class. Contains functions for training and saving checkpoints. """
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders
        self.settings = settings
        self.epoch = 0
        self.stats = OrderedDict()
        self.batch_timer = AverageMeter()
        self.data_timer = AverageMeter()
        if settings.local_rank == 0:
            self.tensorboard_writer = TensorboardWriter(settings.env.tensorboard_dir, 'logs')

    def _set_debug(self):
        pass

    def _get_stats(self):
        return self.stats

    def _print_stats(self, i, epoch, loader, batch_timer, data_timer):
        if self.settings.local_rank != 0:
            return
        print_str = '[%d: %d/%d] ' % (epoch, i, len(loader))
        for name, val in self.stats.items():
            if 'Loss' in name:
                print_str += '%s: %.4f, ' % (name, val.val)
        print_str += 'Data: %.3f, Batch: %.3f' % (data_timer.avg, batch_timer.avg)
        print(print_str)
        if self.settings.env.tensorboard_dir is not None:
            for name, val in self.stats.items():
                self.tensorboard_writer.add_scalar(name, val.val, self.epoch * len(loader) + i)


class Trainer(BaseTrainer):
    """ Trainer class for training the SMAT model with Domain Adaptation. """
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
        self._set_debug()
        self.loader = loaders[0]
        self.domain_loss_fn = nn.BCEWithLogitsLoss()

    def train_epoch(self):
        self.actor.train()
        self.stats = OrderedDict({
            'Loss/total': StatValue(), 'Loss/track': StatValue(), 'Loss/domain': StatValue(),
        })
        if hasattr(self.actor, 'loss_weight') and self.actor.loss_weight is not None:
            for k in self.actor.loss_weight.keys():
                self.stats['Loss/' + k] = StatValue()
            if 'avg_iou' in self.actor.loss_weight.keys():
                 self.stats['IoU'] = StatValue()
        self.batch_timer.reset(); self.data_timer.reset()
        end = time.time()

        for i, data in enumerate(self.loader):
            self.data_timer.update(time.time() - end)
            data = {k: v.to(self.settings.device) for k, v in data.items() if isinstance(v, torch.Tensor)}

            loss, stats_new = self.actor(data)
            
            domain_pred_s = stats_new.pop('domain_pred_s')
            domain_pred_t = stats_new.pop('domain_pred_t')
            domain_pred = torch.cat((domain_pred_s, domain_pred_t), 0)

            batch_size = domain_pred_s.size(0)
            source_labels = torch.zeros(batch_size, 1, device=self.settings.device)
            target_labels = torch.ones(batch_size, 1, device=self.settings.device)
            domain_labels = torch.cat((source_labels, target_labels), 0)
            
            domain_loss = self.domain_loss_fn(domain_pred, domain_labels)

            tracking_loss = loss
            # Note: Accessing cfg through self.settings.cfg, assuming it's attached.
            domain_weight = self.actor.cfg.TRAIN.get('DOMAIN_WEIGHT', 0.1) 
            total_loss = tracking_loss + domain_weight * domain_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.actor.cfg.TRAIN.GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor.cfg.TRAIN.GRAD_CLIP_NORM)
            self.optimizer.step()

            self.batch_timer.update(time.time() - end)
            end = time.time()
            
            self.stats['Loss/total'].update(total_loss.item())
            self.stats['Loss/track'].update(tracking_loss.item())
            self.stats['Loss/domain'].update(domain_loss.item())
            for k, v in stats_new.items():
                if k in self.stats: self.stats[k].update(v)
            if self.settings.print_interval > 0 and i % self.settings.print_interval == 0:
                self._print_stats(i, self.epoch, self.loader, self.batch_timer, self.data_timer)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def run(self):
        """ Runs the training process. """
        for epoch in range(self.actor.cfg.TRAIN.EPOCHS):
            self.epoch = epoch
            if self.settings.distributed:
                self.loader.sampler.set_epoch(epoch)
            self.train_epoch()
            if (epoch + 1) % self.actor.cfg.TRAIN.SAVE_INTERVAL == 0:
                if self.settings.local_rank == 0:
                    self.actor.save_checkpoint(self.settings.env.workspace_dir, epoch=epoch)
        if self.settings.local_rank == 0:
            self.actor.save_checkpoint(self.settings.env.workspace_dir, epoch='final')

def run(settings):
    """
    The main entry point for the training script.
    This function is called by `run_training.py`.
    """
    settings.description = 'Training script for MobileVIT-Track with Domain Adaptation'

    # Update the default cfg with the experiment-specific yaml file
    update_config_from_file(settings.cfg_file, cfg)

    # Set device
    device_str = getattr(cfg.TEST, 'DEVICE', 'cuda')
    device = torch.device(device_str)
    settings.device = device

    # Create network
    net = build_mobilevitv2_track(cfg)
    net.to(device)

    # Set objective functions
    objective = {'box': giou_loss, 'iou': nn.L1Loss()}
    loss_weight = {'box': cfg.TRAIN.GIOU_WEIGHT, 'iou': cfg.TRAIN.L1_WEIGHT}

    # Create actor
    actor = MobileViTTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    # Optimizer
    optimizer = optim.AdamW(actor.net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.LR_DROP_EPOCH, gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE)

    # --- 已修正：将 cfg 对象作为参数传递 ---
    # Create data loader
    loaders = create_loader('train', settings, cfg, STARKProcessing)

    # Create trainer and run training
    trainer = Trainer(actor, loaders, optimizer, settings, lr_scheduler)
    trainer.run()
