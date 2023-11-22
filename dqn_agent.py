import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from inside_agent import InsideAgentForInitState, InsideAgentForAction
from outside_agent import OutsideStateModel, OutsideComModel
from environment import get_rule


def state2str(state):
    now_str = ""
    for s in state:
        now_str += str(s.item())
    return now_str

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, state_range, action_range, vocab_size, device):
        super(Qnet, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_range = state_range
        self.action_range = action_range
        self.vocab_size = vocab_size
        self.device = device

        self.inside_state_net = InsideAgentForInitState(n_digits=state_dim, n_states_per_digit=state_range,
                                                       vocab_size=vocab_size)
        self.inside_action_net = InsideAgentForAction(n_digits=action_dim, n_states_per_digit=action_range,
                                                     vocab_size=vocab_size)
        self.outside_state_net = OutsideStateModel(output_dim=state_dim, state_dim=state_range, hidden_dim=hidden_dim,
                                                  vocab_size=vocab_size)
        self.outside_com_net = OutsideComModel(input_dim=state_dim, hidden_dim=hidden_dim, vocab_size=vocab_size)

    def forward(self, state):
        state = state.long()
        # print(f"qnet forward state = {state}")
        inside_symbol_dist = torch.softmax(self.inside_state_net(state), dim=-1)
        # print(f"inside_symbol_dist.shape = {inside_symbol_dist.shape}")
        inside_symbol_idx = torch.tensor(torch.argmax(inside_symbol_dist, dim=-1, keepdim=True), dtype=torch.long)
        # print(f"inside_symbol_idx = {inside_symbol_idx}")
        if len(inside_symbol_idx.shape) == 1:
            inside_symbol_idx = torch.unsqueeze(inside_symbol_idx, dim=0)
        # print(f"inside_symbol_idx.shape = {inside_symbol_idx.shape}")
        outside_state_dist = self.outside_state_net(inside_symbol_idx)
        outside_state_idx = torch.argmax(outside_state_dist, dim=-1) # (Batch_size, state_dim)
        outside_state_str = [state2str(s) for s in outside_state_idx]
        rule = get_rule(space_dim=self.state_dim, state_dim=self.state_range)
        goal_state = [torch.tensor(rule[s]) for s in outside_state_str]
        goal_state = torch.stack(goal_state, dim=0).to(self.device)
        # print(f"goal_state = {goal_state}")
        outside_symbol_dist = self.outside_com_net(goal_state)
        outside_symbol_idx = torch.argmax(outside_symbol_dist, dim=-1)
        outside_symbol_onehot = F.one_hot(outside_symbol_idx, num_classes=self.vocab_size).float().to(self.device)
        # print(f"outside_symbol_onehot = {outside_symbol_onehot}")
        inside_action_dist = self.inside_action_net(outside_symbol_onehot)

        return inside_action_dist


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, state_range, action_range, vocab_size, lr, gamma, epsilon,
                 target_update, device):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_range = state_range
        self.action_range = action_range
        self.vocab_size = vocab_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.q_net = Qnet(state_dim, hidden_dim, action_dim, state_range, action_range, vocab_size, device).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim, state_range, action_range, vocab_size, device).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.count = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = [np.random.randint(self.action_range) for i in range(self.action_dim)]
            action = np.array(action)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action_dist = self.q_net(state)
            action = torch.argmax(action_dist, dim=-1).numpy()
        return action

    def update(self, data):
        states = torch.tensor(data['state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(data['action'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(data['reward'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(data['next_state'], dtype=torch.float).to(self.device)
        dones = torch.tensor(data['done'], dtype=torch.float).to(self.device)

        q_values = self.q_net(states) # (Batch_size, action_dim, action_range)
        q_values = q_values.gather(2, actions.unsqueeze(2))
        # print(f"q_values.shape = {q_values.shape}")

        next_q_values = self.target_q_net(next_states) # (Batch_size, action_dim, action_range)
        max_next_q_values = torch.max(next_q_values, dim=-1) # (Batch_size, aciton_dim)
        # print(f"max_next_q_values.shape = {(self.gamma * max_next_q_values[0]).shape}")
        rewards = rewards.unsqueeze(1).repeat(1, 3)
        # print(f"rewards.shape = {rewards.shape}")
        q_targets = rewards + self.gamma * max_next_q_values[0] # * (1-dones)
        # print(f"q_values.shape = {q_values.shape}")
        # print(f"q_targets.shape = {q_targets.shape}")
        dqn_loss = torch.mean(F.mse_loss(q_values.view(-1), q_targets.view(-1)))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1








