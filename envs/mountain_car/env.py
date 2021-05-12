import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
from envs.utils import goal_distance, goal_distance_obs
import copy

class MountaincarEnv:
    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        # self.max_speed = 0.27
        self.max_speed = 0.07
        self.goal_pos = self.generate_goal()
        self.power = 0.0015
        self.distance_threshold = 0.05
        self.last_obs = None

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.goal_pos = self.generate_goal()

        self.state = np.array([-0.5, 0])
        # return np.array(self.state)
        self.last_obs = {'desired_goal': np.array(self.goal_pos), 
                'achieved_goal': np.array([self.state[0]]), 
                'observation': np.array(self.state)}

        return {'desired_goal': np.array(self.goal_pos), 
                'achieved_goal': np.array([self.state[0]]), 
                'observation': np.array(self.state)}

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power - 0.0025 * math.cos(3*position)
        if (velocity > self.max_speed):
            velocity = self.max_speed
        if (velocity < -self.max_speed):
            velocity = -self.max_speed
        position += velocity

        if (position > self.max_position):
            position = self.max_position
        if (position < self.min_position):
            position = self.min_position
        
        if (position == self.min_position and velocity < 0):
            velocity = 0
        self.state = np.array([position, velocity])

        self.last_obs = {'desired_goal': np.array(self.goal_pos), 
                'achieved_goal': np.array([self.state[0]]), 
                'observation': np.array(self.state)}

        # here should be changed
        # reward = 0
        # if done:
        #     reward = 100.0
        # reward -= math.pow(action[0], 2)*0.1
        reward = self.compute_reward(self.last_obs['achieved_goal'], self.last_obs['desired_goal'])
        dis = self.compute_distance(self.last_obs['achieved_goal'], self.last_obs['desired_goal'])

        done = False
        if dis <= self.distance_threshold:
            succ = 1.0
            done = True
            # print('done once')
            # print(self.last_obs['desired_goal'])
        else:
            succ = 0
        info = {'Distance': dis,
                'Success': succ}
        return self.last_obs.copy(), reward, False, info
    
    def get_obs(self):
        return self.last_obs

    def compute_reward(self, achieved, goal):
        dis = goal_distance(achieved[0], goal)
        return -1.0 if dis>self.distance_threshold else 0

    def compute_distance(self, achieved, goal):
        return np.sqrt(np.sum(np.square(achieved-goal)))
    
    def generate_goal(self):
        return np.array([0.5 + random.uniform(-0.05, 0.05)]).copy()
        # return np.array([0.5]).copy()

    @property
    def goal(self):
        return self.goal_pos.copy()

    @goal.setter
    def goal(self, value):
        self.goal_pos = value.copy()
        self.last_obs = {'desired_goal': np.array(self.goal_pos), 
                'achieved_goal': self.last_obs['achieved_goal'].copy(), 
                'observation': self.last_obs['observation'].copy()}
