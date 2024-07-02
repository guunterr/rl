import os
os.environ["ROCM_PATH"] = "/opt/rocm"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from utils import *
import time
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper, Shuffler # type: ignore

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def train_loop(dataloader: DataLoader, model, policy_loss_fn, value_loss_fn, optimizer,epoch,save=True,print_progress=True,print_counter=10, save_counter=100):
    losses = []
    model.train()
    start_time = time.time()
    for batch, (b,m,r) in enumerate(dataloader):
        (b,m,r) = (b.to(device),m.to(device),r.to(device))
        (p, v) = model(b.to(device))
        policy_loss = policy_loss_fn(p, m)
        value_loss = value_loss_fn(v, r)
        loss = policy_loss + value_loss
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        if print_progress and (batch+1)%print_counter == 0:
            print(f"policy loss = {policy_loss:.5f}, value_loss = {value_loss:.5f}, batch = {batch+1}, time/10000samples = {(time.time() - start_time)*(10000/(print_counter*dataloader.batch_size)):.4f}")
            start_time = time.time()
        if save and (batch+1)%save_counter == 0:
            torch.save({
                'epoch' : epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
            }, f"checkpoints/1/checkpoint_{epoch}_{batch+1}.pt")
    if save: 
        torch.save({
            'epoch' : epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, f"checkpoints/{epoch}/checkpoint_{epoch}_end.pt")


