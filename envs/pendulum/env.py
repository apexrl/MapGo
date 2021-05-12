import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import random
from envs.utils import goal_distance, goal_distance_obs

class PendulumEnv:
    def __init__(self):
        self.max_speed=8   # 最大角速度：theta dot
        self.max_torque=2. # 最大力矩
        self.dt=.05 
        self.viewer = None

        # add
        self.distance_threshold = 0.1
        self.goal = self.generate_goal()

        high = np.array([1., 1., self.max_speed]) # 这里的前两项是 cos 和 sin 值
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        # 动作空间： (-2, 2) 
        # 观察空间： ([-1,-1,-8],[1,1,8])

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def angle_normalize(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def get_obs_inside(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
        # return self.last_obs
    
    def get_obs(self):
        return self.last_obs

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self.goal = self.generate_goal()
        self.last_obs = {'desired_goal': np.array(self.goal), 
                'achieved_goal': np.array(self.get_obs_inside()[1:3]), 
                'observation': np.array(self.get_obs_inside())}

        return self.last_obs.copy()
    
    def compute_reward(self, achieved, goal):
        dis = goal_distance(achieved[0], goal)
        return -1.0 if dis>self.distance_threshold else 0
    
    def compute_reward_direct(self, achieved, goal):
        dis = goal_distance(achieved, goal)
        return -1.0 if dis>self.distance_threshold else 0

    def compute_distance(self, achieved, goal):
        return np.sqrt(np.sum(np.square(achieved-goal)))

    def generate_goal(self):
        return np.array([random.uniform(-0.2, 0.2), random.uniform(-1, 1)])

    def step(self,u): # u 是 action
        u = u * 2
        th, thdot = self.state # th := theta

        g = 10. # 重力加速度
        m = 1.  # 质量 
        l = 1.  # 长度
        dt = self.dt # 采样时间

        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        self.last_u = u # for rendering
        # costs = self.angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        
        self.state = np.array([newth, newthdot])

        self.last_obs = {'desired_goal': np.array(self.goal), 
                'achieved_goal': np.array(self.get_obs_inside()[1:3]), 
                'observation': np.array(self.get_obs_inside())}

        reward = self.compute_reward_direct(self.last_obs['achieved_goal'], self.last_obs['desired_goal'])
        dis = self.compute_distance(self.last_obs['achieved_goal'], self.last_obs['desired_goal'])
        if dis <= self.distance_threshold:
            succ = 1.0
            done = True
        else:
            succ = 0
        info = {'Distance': dis,
                'Success': succ}
        return self.last_obs.copy(), reward, False, info
