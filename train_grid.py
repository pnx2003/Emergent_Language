import torch
import random
import numpy as np

from tqdm import tqdm

from config import parse_args
from environment import Wire3Env
from dqn_agent import ReplayBuffer, DQN
import matplotlib.pyplot as plt
from gridworld import GridWorld

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = GridWorld()

    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    replay_buffer = ReplayBuffer(args.capacity)

    state_dim = env.world_size * env.world_size * 4 + env.world_size * env.world_size
    state_range = 2

    action_dim = 1
    action_range = 4

    agent = DQN(state_dim=state_dim, hidden_dim=args.hidden_dim, action_dim=action_dim,
                state_range=state_range, action_range=action_range, vocab_size=args.vocab_size,
                lr=args.lr, gamma=args.gamma, epsilon=args.epsilon, target_update=args.target_update,
                device=device)
    
    
    return_list = []
    pic_return_list = []
    step_list = []
    x = []
    cnt = 0
    for i in range(10):
        with tqdm(total=int(args.num_episode / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episode / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                cnt += 1
                x.append(cnt)
                while not done:
                    action = agent.take_action(state, env.goal_state)
                    if len(action.shape) == 2:
                        action = np.squeeze(action, axis=0)
                        
                    next_pos, reward, done = env.step(action[0])
                    # if i_episode == 10:
                    #     print(action, next_state, reward)
                    next_state = np.concatenate([np.zeros((4,4)).flatten() ,env.wall.flatten()])
                    next_state[next_pos[0] * env.world_size + next_pos[1]] = 1
                    replay_buffer.add(state, action, reward, next_state, env.goal_state, done)
                    state = next_state
                    
                    episode_return += reward

                    if replay_buffer.size() > args.minimal_size:
                        b_s, b_a, b_r, b_ns, b_g, b_d = replay_buffer.sample(args.batch_size)
                        data = {
                            'state':       b_s,
                            'action':      b_a,
                            'next_state':  b_ns,
                            'reward':      b_r,
                            'goal_state':  b_g,
                            'done':        b_d,
                        }
                        agent.update(data)
                return_list.append(episode_return)
                pic_return_list.append(np.mean(return_list[-10:]))
                step_list.append(env.now_step)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':  '%d' % (args.num_episode / 10 * i + i_episode + 1),
                        'return':   '%.3f' % np.mean(return_list[-10:]),
                    })
                    # with open('debug/output1.txt', 'a') as f:
                    #     f.write(f'{args.num_episode / 10 * i + i_episode + 1}: {np.mean(return_list[-10:])}')
                    #     f.write('\n')
                pbar.update(1)

    plt.plot(x, pic_return_list)
    plt.show()