import math
import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Dict, Box
from gym.envs.robotics import rotations, utils

import numpy as np
from envs.utils import goal_distance, goal_distance_obs
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

class ReacherThreeEnvInner(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, threshold=0.02):
        self.threshold = threshold
        self.ac_dim = 3
        self.state_dim = 11
        self.goal_dim = 2
        self._max_episode_steps = 50
        self.goal = np.array([0.0,0.0])
        self.name = "reacher_three"
        self.object = False

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/reacher_three.xml', 2)

    def _is_success(self, achieved_goal, goal):
        d = np.linalg.norm(achieved_goal-goal)
        return (d < self.threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal, info):
        d = np.linalg.norm(achieved_goal-goal, axis=-1)
        return -(d > self.threshold).astype(np.float32)

    def save_state(self):
        return self.sim.get_state()

    def restore_state(self, state):
        self.sim.set_state(state)
        self.sim.forward()

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        # reward_dist = - np.linalg.norm(vec)
        # reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], self.goal, None)
        done = False
        info = {"is_success": self._is_success(obs["achieved_goal"], obs["desired_goal"])}
        return obs, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.25, high=.25, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        obs = {}
        obs["observation"] = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip")[:2]
        ])
        obs["achieved_goal"] = self.get_body_com("fingertip")[:2]
        obs["desired_goal"] = self.goal

        return obs

    def get_goal_from_state(self, state):
        if len(state.shape) == 1:
            return state[-2:]
        else:
            return state[..., -2:]

    def batch_goal_achieved(self, states, goals, tol=0.02):
        states = self.get_goal_from_state(states)
        dists = np.linalg.norm(states-goals, axis=-1)
        return (dists < tol).astype(int)

class ReacherThreeEnv:
    def __init__(self):
        self.internal_world = ReacherThreeEnvInner()

        self.action_space = self.internal_world.action_space
        # self.goal_space = self.internal_world.goal_space
        self.observation_space = self.internal_world.observation_space['observation']

        self.distance_threshold = 0.02

        self.seed()
        self.reset()    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        obs = self.internal_world.reset()
        obs['observation'] = obs['observation']
        obs['achieved_goal'] = np.array(obs['achieved_goal'])
        self.goal_ = obs['desired_goal']
        return obs

    def step(self, action):
        obs, reward, done, info = self.internal_world.step(action)
        obs['desired_goal'] = self.goal_
        # obs['achieved_goal'] = obs['achieved_goal']
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
        # obs['achieved_goal'] = obs['achieved_goal']
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
        temp_env = ModifiedFP()
        obs = temp_env.reset()
        sampled_desired_goals = obs['desired_goal'].copy()
        del temp_env
        return sampled_desired_goals

    @property
    def goal(self):
        return self.goal_.copy()

    @goal.setter
    def goal(self, value):
        self.goal_ = value