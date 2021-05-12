import math
import multiworld
import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Dict, Box
import numpy as np
from envs.utils import goal_distance, goal_distance_obs

class Ant_Locomotion:
    def __init__(self):
        multiworld.register_all_envs()
        self.internal_world = gym.make('AntXY-v0')
        self.internal_world.include_contact_forces_in_state = False
        
        self.distance_threshold = 0.1
        self.goal_limit = 2
        self.goal_low = -self.goal_limit * np.ones(2)
        self.goal_high = self.goal_limit * np.ones(2)
        self.goal_ = None
        self.goal_space = Box(self.goal_low, self.goal_high)
        self.action_space = self.internal_world.action_space
        self.observation_space_low = np.array([-np.inf for _ in range(29)])
        self.observation_space_high = np.array([np.inf for _ in range(29)])
        self.observation_space = Box(self.observation_space_low, self.observation_space_high)
        
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        obs = self.internal_world.reset()
        obs['observation'] = obs['observation'][0:29]
        self.goal_ = self.goal_space.sample()
        obs['desired_goal'] = self.goal_
        return obs

    def step(self, action):
        obs, reward, done, info = self.internal_world.step(action)
        obs['desired_goal'] = self.goal_
        reward = self.compute_reward_direct(obs['achieved_goal'], obs['desired_goal'])
        dis = self.compute_distance(obs['achieved_goal'], obs['desired_goal'])
        if dis <= self.distance_threshold:
            succ = 1
        else:
            succ = 0
        info = {'Distance': dis,
                'Success': succ}
        return obs, reward, False, info

    def get_obs(self):
        obs = self.internal_world._get_obs()
        obs['desired_goal'] = self.goal_.copy()
        return obs

    def compute_reward(self, achieved, goal):
        dis = goal_distance(achieved[0], goal)
        return -1.0 if dis>self.distance_threshold else 0
    
    def compute_reward_direct(self, achieved, goal):
        dis = goal_distance(achieved, goal)
        return -1.0 if dis>self.distance_threshold else 0

    def compute_distance(self, achieved, goal):
        return np.sqrt(np.sum(np.square(achieved - goal)))

    def generate_goal(self):
        return self.goal_space.sample()

    @property
    def goal(self):
        return self.goal_.copy()

    @goal.setter
    def goal(self, value):
        self.goal_ = value