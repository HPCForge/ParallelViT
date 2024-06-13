import os
import time
import functools
import torch
import torch.nn.functional as F
import torch.optim as optim

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)

from models.vit import VisionTransformer
from utils.dist_utils import setup, cleanup, is_leader_process
from data import ConcatData, BubbleML

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)

fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

FILES = ['/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-90.hdf5',  
  '/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-94.hdf5',  
  '/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-96.hdf5',
#   '/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-98.hdf5',
#   '/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-100.hdf5',
#   '/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-102.hdf5',
#   '/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-104.hdf5',
#   '/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-108.hdf5',
#   '/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-110.hdf5',
#   '/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-92.hdf5'
]



def fsdp_main():
    setup()

    model = VisionTransformer(img_size=512, 
                              patch_size=16,
                              in_channels=4,
                              embed_dim=4096,
                              depth=12,
                              num_heads=32,
    )
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    train_dataset = ConcatData(BubbleML(filename=f, norm='minmax') for f in FILES[:-1])
    data_min, data_max = train_dataset.minmax_normalize()
    val_dataset = BubbleML(filename=FILES[-1], norm='minmax')
    val_dataset.minmax_normalize(data_min, data_max)

    train_sampler = DistributedSampler(train_dataset, 
                                 rank=rank, 
                                 num_replicas=world_size, 
                                 shuffle=True)
    val_sampler = DistributedSampler(val_dataset,
                                 rank=rank,
                                 num_replicas=world_size,
                                 shuffle=False)
    
    train_dataloader = DataLoader(train_dataset, 
                            sampler=train_sampler, 
                            batch_size=4,  
                            pin_memory=True,
                            num_workers=1, 
                            prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset,
                            sampler=val_sampler,
                            batch_size=4,
                            pin_memory=True,
                            num_workers=1, 
                            prefetch_factor=2)
    
    torch.cuda.set_device(local_rank)

    # auto_wrap_policy = functools.partial(
    #     size_based_auto_wrap_policy, min_num_params=100
    # )

    model = FSDP(model,
        auto_wrap_policy=size_based_auto_wrap_policy,
        mixed_precision=fpSixteen,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    for epoch in range(10):
        model.train()
        t0 = time.perf_counter()
        fsdp_train_loss = torch.zeros(2).to(local_rank)
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(torch.float16).to(local_rank)
            targets = targets.to(torch.float16).to(local_rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            fsdp_train_loss[0] += loss.item()
            fsdp_train_loss[1] += len(inputs)
            
        dist.all_reduce(fsdp_train_loss, op=dist.ReduceOp.SUM)
        if is_leader_process():
                print(f"Epoch {epoch} Train Loss {fsdp_train_loss[0]/fsdp_train_loss[1]} Time {time.perf_counter()-t0}")
        model.eval()
        fsdp_val_loss = torch.zeros(2).to(local_rank)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_dataloader):
                inputs = inputs.to(torch.float16).to(local_rank)
                targets = targets.to(torch.float16).to(local_rank)
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                fsdp_val_loss[0] += loss.item()
                fsdp_val_loss[1] += len(inputs)
            dist.all_reduce(fsdp_val_loss, op=dist.ReduceOp.SUM)
            curr_val_loss = fsdp_val_loss[0]/fsdp_val_loss[1]
            if curr_val_loss < best_val_loss:
                best_val_loss = curr_val_loss
                # torch.save(model.state_dict(), "best_model.pth")
        scheduler.step()
        if is_leader_process():
                print(f"Epoch {epoch} Val Loss {curr_val_loss}")
        
    dist.barrier()
    cleanup()

if __name__ == '__main__':
     fsdp_main()



