from environment import Wire3Env
from dqn_agent import ReplayBuffer, DQN

import torch
import numpy as np
import random
import argparse
from tqdm import tqdm





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=5000)
    parser.add_argument("--minimal-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--target-update", type=int, default=5)
    parser.add_argument("--num-episode", type=int, default=10000)

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = "cpu"

    env = Wire3Env()

    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    replay_buffer = ReplayBuffer(args.capacity)

    state_dim = env.observation_space.shape[0]
    state_range = env.observation_space[0].n

    action_dim = env.action_space.shape[0]
    action_range = env.action_space[0].n

    agent = DQN(state_dim=state_dim, hidden_dim=args.hidden_dim, action_dim=action_dim,
                state_range=state_range, action_range=action_range, vocab_size=args.vocab_size,
                lr=args.lr, gamma=args.gamma, epsilon=args.epsilon, target_update=args.target_update,
                device=device)


    return_list = []
    for i in range(10):
        with tqdm(total=int(args.num_episode / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episode / 10)):
                episode_return = 0
                state = env.remake()
                done = False
                while not done:
                    action = agent.take_action(state)
                    if len(action.shape) == 2:
                        action = np.squeeze(action, axis=0)
                    next_state, reward, done = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    if replay_buffer.size() > args.minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(args.batch_size)
                        data = {
                            'state': b_s,
                            'action': b_a,
                            'next_state': b_ns,
                            'reward': b_r,
                            'done': b_d
                        }
                        agent.update(data)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (args.num_episode / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

