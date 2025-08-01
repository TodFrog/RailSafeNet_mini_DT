import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import os
import wandb
from tqdm import tqdm
import time
import copy

# ✅ [최종 해결책] wandb가 DDP의 spawn과 충돌하지 않도록 시작 방식을 'thread'로 강제합니다.
# 이 코드는 모든 import 구문보다 위에 있는 것이 가장 안전합니다.
os.environ["WANDB_START_METHOD"] = "thread"

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from scripts.dataloader_SegFormer import CustomDataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.optim import Adam, Adagrad, AdamW

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356' 
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_cleanup():
    dist.destroy_process_group()

PATH_JPGS = "/home/mmc-server3/Server/Datasets/rs19_val/jpgs/rs19_val"
PATH_MASKS = "/home/mmc-server3/Server/Datasets/rs19_val/uint8/rs19_val"
PATH_MODELS = "RailNet_DT/models"

def create_model(output_channels=1):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        num_labels=output_channels,
        ignore_mismatched_sizes=True
    )
    return model

def train(rank, world_size, model, config, optimizer, criterion, scheduler):
    device = rank
    per_gpu_batch_size = config['batch_size'] // world_size
    
    image_processor = SegformerImageProcessor(size={"height": config['image_size'], "width": config['image_size']})
    train_dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_processor, [config['image_size'], config['image_size']], 'Train', config['outs'])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=per_gpu_batch_size, shuffle=False, drop_last=True, sampler=train_sampler, num_workers=4)

    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        total_train_loss = 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", disable=(rank != 0))
        for inputs, masks in train_loader_tqdm:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            logits = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = torch.tensor(total_train_loss / len(train_loader), device=device)
        dist.all_reduce(avg_train_loss, op=dist.ReduceOp.AVG)

        if config['scheduler'] == 'LinearLR': scheduler.step()
        
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{config['epochs']}: Train loss: {avg_train_loss.item():.4f} | lr: {current_lr:.6f}")
            wandb.log({"train_loss": avg_train_loss.item(), "lr": current_lr, "epoch": epoch})

def ddp_train_worker(rank, world_size, config_dict, run_id):
    ddp_setup(rank, world_size)
    
    if rank == 0:
        # 부모의 run에 재접속하는 로직은 그대로 유지합니다.
        wandb.init(id=run_id, resume="must")
    
    model = create_model(config_dict['outs']).to(rank)
    model = DDP(model, device_ids=[rank])

    if config_dict['optimizer'] == 'adam': optimizer = Adam(model.parameters(), lr=config_dict['learning_rate'])
    elif config_dict['optimizer'] == 'adagrad': optimizer = Adagrad(model.parameters(), lr=config_dict['learning_rate'])
    
    if config_dict['scheduler'] == 'LinearLR': scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
    else: scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    loss_function = nn.CrossEntropyLoss(ignore_index=255)
    
    train(rank, world_size, model, config_dict, optimizer, loss_function, scheduler)
    
    if rank == 0:
        wandb.finish()

    ddp_cleanup()

def main_launcher():
    with wandb.init() as run:
        config = wandb.config
        world_size = torch.cuda.device_count()
        run_id = run.id
        
        config_dict = dict(config)
        
        assert config.batch_size % world_size == 0, "Sweep의 배치 사이즈는 GPU 개수로 나누어 떨어져야 합니다."
        
        if not os.path.exists(PATH_MODELS):
            os.makedirs(PATH_MODELS)
        
        print(f"Sweep 시작: {run.name} / {world_size}개의 GPU / 전체 배치 사이즈: {config.batch_size}")
        
        mp.spawn(ddp_train_worker, args=(world_size, config_dict, run_id), nprocs=world_size)

if __name__ == "__main__":
    sweep_config = {
        'method': 'random',
        'metric': { 'name': 'train_loss', 'goal': 'minimize' },
        'parameters': {
            'epochs': {'value': 50},
            'learning_rate': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.001},
            'optimizer': {'values': ['adam', 'adagrad']},
            'scheduler': {'values': ['LinearLR']},
            'batch_size': {'values': [16, 32]},
            'image_size': {'value': 1024},
            'outs': {'value': 19}
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="DP_train_full")
    wandb.agent(sweep_id, function=main_launcher, count=10)