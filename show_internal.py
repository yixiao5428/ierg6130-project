import argparse
import os
import numpy as np
import torch

from core.network import ActorCritic

if __name__ == "__main__":
    trainer_path = "/tmp/ierg6130-project/PPO/checkpoint-iter1000.pkl"
    save_path = trainer_path
    model = ActorCritic(torch.Size([512]), 6)
    if os.path.isfile(save_path):
        state_dict = torch.load(
            save_path,
            torch.device('cpu') if not torch.cuda.is_available() else None
        )
        model.load_state_dict(state_dict["model"])

    fc1_w = model.state_dict()['fc1.weight'] 
    importance = torch.abs(torch.sum(fc1_w, axis=0))
    importance_sorted = torch.sort(importance)
    print(importance_sorted)
    print(torch.max(torch.abs(importance)), torch.min(torch.abs(importance)))
