import argparse
import os
import numpy as np
import torch

# from core.a2c_trainer import A2CTrainer, a2c_config
# from core.ppo_trainer import PPOTrainer, ppo_config
# from core.utils import verify_log_dir, pretty_print, Timer, evaluate, \
#     summary, save_progress, FrameStackTensor, step_envs
from core.network import ActorCritic
from train import make_envs, parse_args_for_train

if __name__ == "__main__":
    # args = parse_args_for_train()

    # algo = args.algo
    # if algo == "PPO":
    #     config = ppo_config
    # elif algo == "A2C":
    #     config = a2c_config
    # else:
    #     raise ValueError("args.algo must in [PPO, A2C]")
    # config.num_envs = args.num_envs
    # config.num_envs = args.num_envs


    # # Seed the environments and setup torch
    # seed = args.seed
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    # torch.set_num_threads(1)

    # # Clean log directory
    # log_dir = verify_log_dir(args.log_dir, algo)

    # # Create vectorized environments
    # num_envs = args.num_envs
    # env_id = args.env_id
    # envs = make_envs(env_id, log_dir, num_envs, False)
    # eval_envs = make_envs(env_id, log_dir, num_envs, False)

    # # Setup trainer
    # test = False
    # frame_stack = 4
    # if algo == "PPO":
    #     trainer = PPOTrainer(envs, config, frame_stack, _test=test)
    # else:
    #     trainer = A2CTrainer(envs, config, frame_stack, _test=test)

    trainer_path = "/tmp/ierg6130-project/PPO/checkpoint-iter1000.pkl"
    save_path = trainer_path
    model = ActorCritic(torch.Size([512]), 6)
    if os.path.isfile(save_path):
        state_dict = torch.load(
            save_path,
            torch.device('cpu') if not torch.cuda.is_available() else None
        )
        model.load_state_dict(state_dict["model"])

    print(model.state_dict()['fc1.weight'])
