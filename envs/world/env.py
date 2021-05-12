import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
from envs.utils import goal_distance, goal_distance_obs
import copy

class WorldEnv:
    def __init__(self):
        self.min_action = [-1.0, -1.0]
        self.max_action = [1.0, 1.0]
        self.min_position = [0, 0]
        self.max_position = [20, 20]

        self.goal_pos = self.generate_goal()
        self.distance_threshold = 0.15

        self.low_state = np.array(self.min_position)
        self.high_state = np.array(self.max_position)

        self.action_space = spaces.Box(low=np.array(self.min_action), high=np.array(self.max_action))
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def generate_goal(self):
        return np.array([18+random.uniform(-0.5, 0.5), 18+random.uniform(-0.5, 0.5)])

    def reset(self):
        self.goal_pos = self.generate_goal()

        self.state = np.array([0, 0])
        self.last_obs = {'desired_goal': np.array(self.goal_pos), 
                'achieved_goal': np.array(self.state), 
                'observation': np.array(self.state)}

        return {'desired_goal': np.array(self.goal_pos), 
                'achieved_goal': np.array(self.state), 
                'observation': np.array(self.state)}

    def step(self, action):
        x = self.state[0]
        y = self.state[1]

        x += action[0]
        y += action[1]
        if x>20:
            x = 20
        if x<0:
            x = 0
        if y>20:
            y = 20
        if y<0:
            y = 0
            
        self.state = np.array([x, y])
        self.last_obs = {'desired_goal': np.array(self.goal_pos), 
                'achieved_goal': np.array(self.state), 
                'observation': np.array(self.state)}
        
        reward = self.compute_reward_direct(self.last_obs['achieved_goal'], self.last_obs['desired_goal'])
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
    
    def compute_reward_direct(self, achieved, goal):
        dis = goal_distance(achieved, goal)
        return -1.0 if dis>self.distance_threshold else 0

    def compute_distance(self, achieved, goal):
        return np.sqrt(np.sum(np.square(achieved-goal)))
    
    @property
    def goal(self):
        return self.goal_pos.copy()

    @goal.setter
    def goal(self, value):
        self.goal_pos = value.copy()
        self.last_obs = {'desired_goal': np.array(self.goal_pos), 
                'achieved_goal': self.last_obs['achieved_goal'].copy(), 
                'observation': self.last_obs['observation'].copy()}
