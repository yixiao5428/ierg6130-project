import argparse
from collections import deque

import gym
import numpy as np
import torch
from competitive_pong.utils import DummyVecEnv, SubprocVecEnv

from core.a2c_trainer import A2CTrainer, a2c_config
from core.ppo_trainer import PPOTrainer, ppo_config
from core.utils import verify_log_dir, pretty_print, Timer, evaluate, \
    summary, save_progress, FrameStackTensor, step_envs


gym.logger.set_level(40)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--algo",
    default="A2C",
    type=str,
    help="(Required) The algorithm you want to run. Must in [PPO, A2C]."
)
parser.add_argument(
    "--log-dir",
    default="/tmp/ierg6130-project/",
    type=str,
    help="The path of directory that you want to store the data to. "
         "Default: /tmp/ierg6130-project/ppo/"
)
parser.add_argument(
    "--num-envs",
    default=15,
    type=int,
    help="The number of parallel environments. Default: 15"
)
parser.add_argument(
    "--learning-rate", "-LR",
    default=5e-4,
    type=float,
    help="The learning rate. Default: 5e-4"
)
parser.add_argument(
    "--seed",
    default=100,
    type=int,
    help="The random seed. Default: 100"
)
parser.add_argument(
    "--max-steps",
    "-N",
    default=1e7,
    type=float,
    help="The random seed. Default: 1e7"
)
parser.add_argument(
    "--env-id",
    default="Pong-ram-v0",
    type=str,
    help="The environment id, Default: Pong-ram-v0"
)
args = parser.parse_args()


def make_envs(env_id, num_envs, asynchronous):
    """

    :param env_id:
    :param num_envs:
    :param asynchronous:
    :return:
    """
    asynchronous = asynchronous and num_envs > 1

    print("Setup environment Pong-ram-v0.")
    envs = [lambda: gym.make(env_id) for i in range(num_envs)]
    envs = SubprocVecEnv(envs) if asynchronous else DummyVecEnv(envs)

    return envs


def train(args):
    """

    :param args:
    :return:
    """
    # Verify algorithm and config
    algo = args.algo
    if algo == "PPO":
        config = ppo_config
    elif algo == "A2C":
        config = a2c_config
    else:
        raise ValueError("args.algo must in [PPO, A2C]")
    config.num_envs = args.num_envs

    # Seed the environments and setup torch
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    # Clean log directory
    log_dir = verify_log_dir(args.log_dir, algo)

    # Create vectorized environments
    num_envs = args.num_envs
    env_id = args.env_id
    envs = make_envs(env_id, num_envs, False)
    eval_envs = make_envs(env_id, num_envs, False)

    # Setup trainer
    test = False
    frame_stack = 4
    if algo == "PPO":
        trainer = PPOTrainer(envs, config, frame_stack, _test=test)
    else:
        trainer = A2CTrainer(envs, config, frame_stack, _test=test)

    # Create a placeholder tensor to help stack frames in 2nd dimension
    # That is turn the observation from shape [num_envs, 1, 84, 84] to
    # [num_envs, 4, 84, 84].
    frame_stack_tensor = FrameStackTensor(
        num_envs, envs.observation_space.shape, frame_stack, config.device)

    # Setup some stats helpers
    episode_rewards = np.zeros([num_envs, 1], dtype=np.float)
    total_episodes = total_steps = iteration = 0
    reward_recorder = deque(maxlen=100)
    episode_length_recorder = deque(maxlen=100)
    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []
    evaluate_stat = {}

    # Start training
    print("Start training!")
    obs = envs.reset()
    frame_stack_tensor.update(obs)
    trainer.rollouts.observations[0].copy_(frame_stack_tensor.get())
    while True:  # Break when total_steps exceeds maximum value
        # ===== Sample Data =====
        with sample_timer:
            for index in range(config.num_steps):
                # Get action
                # [TODO] Get the action
                # Hint:
                #   1. Remember to disable gradient computing
                #   2. trainer.rollouts is a storage containing all data
                #   3. What observation is needed for trainer.compute_action?
                values = None
                actions = None
                action_log_prob = None
                with torch.no_grad():
                    values, actions, action_log_prob = trainer.compute_action(trainer.rollouts.observations[index])

                cpu_actions = actions.view(-1).cpu().numpy()

                # Step the environment
                # (Check step_envs function, you need to implement it)
                obs, reward, done, info, masks, total_episodes, \
                total_steps, episode_rewards = step_envs(
                    cpu_actions, envs, episode_rewards, frame_stack_tensor,
                    reward_recorder, episode_length_recorder, total_steps,
                    total_episodes, config.device, test)

                rewards = torch.from_numpy(
                    reward.astype(np.float32)).view(-1, 1).to(config.device)

                # Store samples
                trainer.rollouts.insert(
                    frame_stack_tensor.get(), actions.view(-1, 1),
                    action_log_prob, values, rewards, masks)

        # ===== Process Samples =====
        with process_timer:
            with torch.no_grad():
                next_value = trainer.compute_values(
                    trainer.rollouts.observations[-1])
            trainer.rollouts.compute_returns(next_value, config.GAMMA)

        # ===== Update Policy =====
        with update_timer:
            policy_loss, value_loss, dist_entropy, total_loss = \
                trainer.update(trainer.rollouts)
            trainer.rollouts.after_update()

        # ===== Evaluate Current Policy =====
        if iteration % config.eval_freq == 0:
            eval_timer = Timer()
            evaluate_rewards, evaluate_lengths = evaluate(
                trainer, eval_envs, frame_stack, 20)
            evaluate_stat = summary(evaluate_rewards, "episode_reward")
            if evaluate_lengths:
                evaluate_stat.update(
                    summary(evaluate_lengths, "episode_length"))
            evaluate_stat.update(dict(
                win_rate=float(
                    sum(np.array(evaluate_rewards) >= 0) / len(
                        evaluate_rewards)),
                evaluate_time=eval_timer.now,
                evaluate_iteration=iteration
            ))

        # ===== Log information =====
        if iteration % config.log_freq == 0:
            stats = dict(
                log_dir=log_dir,
                frame_per_second=int(total_steps / total_timer.now),
                training_episode_reward=summary(reward_recorder,
                                                "episode_reward"),
                training_episode_length=summary(episode_length_recorder,
                                                "episode_length"),
                evaluate_stats=evaluate_stat,
                learning_stats=dict(
                    policy_loss=policy_loss,
                    entropy=dist_entropy,
                    value_loss=value_loss,
                    total_loss=total_loss
                ),
                total_steps=total_steps,
                total_episodes=total_episodes,
                time_stats=dict(
                    sample_time=sample_timer.avg,
                    process_time=process_timer.avg,
                    update_time=update_timer.avg,
                    total_time=total_timer.now,
                    episode_time=sample_timer.avg + process_timer.avg +
                                 update_timer.avg
                ),
                iteration=iteration
            )

        if iteration % config.save_freq == 0:
            trainer_path = trainer.save_w(log_dir, "iter{}".format(iteration))
            progress_path = save_progress(log_dir, progress)
            print("Saved trainer state at <{}>. Saved progress at <{}>.".format(
                trainer_path, progress_path
            ))

        # [TODO] Stop training when total_steps is greater than args.max_steps
        if total_steps > args.max_steps:
            break

        iteration += 1

    trainer.save_w(log_dir, "final")
    envs.close()


if __name__ == '__main__':
    train(args)
