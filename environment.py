import os
import random
import pickle
import gymnasium
import numpy as np

from gymnasium import spaces
from gymnasium.utils import seeding


def generate_state(now_dim, space_dim, state_dim, temp_state, all_state):
    if now_dim == space_dim:
        all_state.append(temp_state.copy())
        return
    for i in range(state_dim):
        temp_state[now_dim] = i
        generate_state(now_dim+1, space_dim, state_dim, temp_state, all_state)

def get_rule(space_dim, state_dim):
    temp_state = np.zeros(shape=space_dim)
    all_state = []
    generate_state(0, space_dim, state_dim, temp_state, all_state)
    all_state = np.array(all_state, dtype=int)
    all_change = []
    for i in range(len(all_state)):
        single_change = [random.randint(0, state_dim-1), random.randint(0, state_dim-1), random.randint(0, state_dim-1)]
        all_change.append(single_change)
    all_change = np.array(all_change, dtype=int)
    all_goal_state = (all_state + all_change) % state_dim
    rule = {}
    for i in range(len(all_state)):
        rule[state2str(all_state[i])] = all_goal_state[i]
    with open("./env/wire3rule.pickle", "wb") as f:
        pickle.dump(rule, f)
    return rule

def state2str(state):
    now_str = ""
    for s in state:
        now_str += str(s)
    return now_str

class Wire3Env(gymnasium.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.action_space = spaces.MultiDiscrete([3,3,3])
        self.observation_space = spaces.MultiDiscrete([3,3,3])
        self.colors = [[255,0,0],[0,255,0],[0,0,255]]
        self.init_state = [random.randint(0,2), random.randint(0,2), random.randint(0,2)]
        self.now_state = self.init_state
        self.now_step = 0
        self.gamma = 0.9
        self.lines = []
        self.viewer = None
        if os.path.exists("./env/wire3rule.pickle"):
            with open("./env/wire3rule.pickle", "rb") as f:
                self.rule = pickle.load(f)
        else:
            self.rule = get_rule(3,3)
        self.goal_state = self.rule[state2str(self.init_state)]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        self.now_state = action
        done = False
        reward = 0
        if np.all(self.now_state == self.goal_state):
            done = True
            reward = np.power(self.gamma, self.now_step)

        self.now_step += 1
        done = True

        return self.now_state, reward, done

    def reset(self):
        self.now_state = self.init_state
        return self.now_state

    def remake(self):
        self.now_step = 0
        self.init_state = [random.randint(0,2), random.randint(0,2), random.randint(0,2)]
        self.goal_state = self.rule[state2str(self.init_state)]
        self.now_state = self.init_state
        return self.init_state

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 400
        from gymnasium.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.lines.append(rendering.FilledPolygon([(100, 100), (100, 300), (115, 100), (115, 300)]))
            self.lines.append(rendering.FilledPolygon([(300, 100), (300, 300), (315, 100), (315, 300)]))
            self.lines.append(rendering.FilledPolygon([(500, 100), (500, 300), (515, 100), (515, 300)]))

            for i in range(len(self.lines)):
                color = self.colors[self.now_state[i]]
                self.lines[i].set_color(color[0], color[1], color[2])
                self.viewer.add_geom(self.lines[i])

        for i in range(len(self.lines)):
            color = self.colors[self.now_state[i]]
            self.lines[i].set_color(color[0], color[1], color[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

if __name__ == "__main__":


    env = Wire3Env()
    print(f"observation_space = {env.observation_space[0].n}")
    print(f"action_space = {env.action_space}")

    # env.reset()
    # print(f"goal_state = {env.goal_state}")
    # while True:
    #     env.render()
    #     action = env.action_space.sample()
    #     s, r, d = env.step(action)
    #     print(f"now state = {s}")
    #     time.sleep(1)
    #     if d:
    #         env.close()
    #         break

