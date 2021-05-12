import math
import multiworld
import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Dict, Box
import numpy as np
from envs.utils import goal_distance, goal_distance_obs

class Half_Cheetah:
    def __init__(self):
        multiworld.register_all_envs()
        self.internal_world = gym.make('HalfCheetahGoal-v0')
        self.max_speed = 6
        self.action_space = self.internal_world.action_space
        self.goal_space = self.internal_world.goal_space
        self.observation_space = self.internal_world.obs_space

        self.distance_threshold = 0.1

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        obs = self.internal_world.reset()
        obs['observation'] = obs['observation']
        obs['achieved_goal'] = np.array([obs['achieved_goal']])
        self.goal_ = obs['desired_goal']
        return obs

    def step(self, action):
        obs, reward, done, info = self.internal_world.step(action)
        obs['desired_goal'] = self.goal_
        obs['achieved_goal'] = np.array([obs['achieved_goal']])
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
        obs['achieved_goal'] = np.array([obs['achieved_goal']])
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