import tensorflow as tf
import numpy as np
import copy
from envs import make_env
from utils.sampler import MatchSampler
from policy.replay_buffer import Trajectory
from envs.utils import goal_distance
from utils.funcs_utils import step_fake, step_fake_batch, compute_reward
import pandas as pd
import pickle
import time

class TrajectoryPool:
    def __init__(self, args, pool_length):
        self.args = args
        self.length = pool_length

        self.pool = []
        self.pool_init_state = []
        self.counter = 0

    def insert(self, trajectory, init_state):
        if self.counter < self.length:
            self.pool.append(trajectory.copy())
            self.pool_init_state.append(init_state.copy())
        else:
            self.pool[self.counter % self.length] = trajectory.copy()
            self.pool_init_state[self.counter % self.length] = init_state.copy()
        self.counter += 1

    def pad(self):
        if self.counter >= self.length:
            return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
        pool = copy.deepcopy(self.pool)
        pool_init_state = copy.deepcopy(self.pool_init_state)
        while len(pool) < self.length:
            pool += copy.deepcopy(self.pool)
            pool_init_state += copy.deepcopy(self.pool_init_state)
        return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

class dynamic_buffer:
    def __init__(self, max_length):
        self.max_dynamic_buffer_size = max_length
        self.dynamic_buffer_p = {'st':0, 'at':0, 'stpo':0}
        self.dynamic_buffer_full = False
        self.dynamic_buffer_number = 0
        self.data = {
            'st': [None for _ in range(self.max_dynamic_buffer_size)],
            'at': [None for _ in range(self.max_dynamic_buffer_size)],
            'stpo': [None for _ in range(self.max_dynamic_buffer_size)]
        }
    
    def add(self, data, data_type):
        # execute append data
        insert_pos = self.dynamic_buffer_p[data_type]
        self.data[data_type][insert_pos] = data
        self.dynamic_buffer_p[data_type] = (self.dynamic_buffer_p[data_type] + 1) % self.max_dynamic_buffer_size

        # execute buffer_number
        if data_type=='stpo':
            if self.dynamic_buffer_full == False:
                self.dynamic_buffer_number += 1
            if not self.dynamic_buffer_full and self.dynamic_buffer_number == self.max_dynamic_buffer_size:
                self.dynamic_buffer_full = True
                print("full!!!!")
                

class HGGLearner:
    def __init__(self, args):
        self.args = args
        self.env = make_env(args)
        self.env_test = make_env(args)

        self.env_List = []
        for i in range(args.episodes):
            self.env_List.append(make_env(args))

        self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
        self.sampler = MatchSampler(args, self.achieved_trajectory_pool)
        
        self.dynamic_buffer = dynamic_buffer(args.buffer_size*100) # need fine tune here
        self.model_loss = []

        try:
            with open("./random_trajs/trajs_0.pkl", 'rb') as f:
                self.hist = pickle.load(f)
        except IOError:
            print("Historical trajectory file not exist.")

        self.hist_trajs = {"eps":[], "obs":[], "goal":[], "action":[]}
        self.current_trajs = {"eps":[], "obs":[], "goal":[], "action":[]}

    def extend_traj(self, env_model, buffer, agent, buffer_fake, extend_length, achieved_init_states=None, achieved_trajectories=None):
        batch_size = 10000
        extend_length = extend_length
        trajs = buffer.sample_batch(batch_size=batch_size)
        start_obs = np.array(trajs['obs'])
        # every time batch size 1000
        for x in range(10):
            obs_batch = start_obs[x*1000:(x+1)*1000] # here obs is [obs, desired_goal]
            # initialize currents.
            currents = []
            pure_obs_batch = []
            desired_goal_batch = []
            for y in range(1000):
                obs = {}
                obs['observation'] = obs_batch[y][:-self.args.env_params['desire_dim']]
                obs['desired_goal'] = obs_batch[y][-self.args.env_params['desire_dim']:]
                obs['achieved_goal'] = obs_batch[y][self.args.env_params['start_in_obs']:self.args.env_params['end_in_obs']]
                currents.append(Trajectory(obs.copy()))
                pure_obs_batch.append(obs['observation'])
                desired_goal_batch.append(obs['desired_goal'])
            pure_obs_batch = np.array(pure_obs_batch)
            desired_goal_batch = np.array(desired_goal_batch)
            
            # extend.
            for y in range(extend_length):
                pi_input = np.concatenate([pure_obs_batch, desired_goal_batch], axis=1)
                actions = agent.step_batch(pi_input, batch_size=1000, explore=True)
                pure_obs_batch = step_fake_batch(
                                    self.args.env_model, pure_obs_batch, actions, self.args.env_params['step_fake_param'], self.args.distance_threshold, args=self.args, batch=1000)
                
                # for every traj, store step
                for z in range(1000):
                    obs = {}
                    obs['observation'] = pure_obs_batch[z]
                    obs['desired_goal'] = desired_goal_batch[z]
                    obs['achieved_goal'] = pure_obs_batch[z][self.args.env_params['start_in_obs']:self.args.env_params['end_in_obs']]
                    reward = compute_reward(obs['desired_goal'], obs['achieved_goal'], self.args.distance_threshold)
                    currents[z].store_step(actions[z], obs.copy(), reward, False)
            for y in range(1000):
                buffer_fake.store_trajectory(currents[y])

    def learn(self, args, env, env_test, agent, buffer, buffer_fake=None, env_model=None, fake=False, test=False):
        self.current_trajs = {"eps":[], "obs":[], "goal":[]}
        self.hist_trajs = {"eps":[], "obs":[], "goal":[]}
        initial_goals = []
        desired_goals = []
        for i in range(args.episodes):
            obs = self.env_List[i].reset()
            goal_a = obs['achieved_goal'].copy()
            goal_d = obs['desired_goal'].copy()
            initial_goals.append(goal_a.copy())
            desired_goals.append(goal_d.copy())

        if args.goal_generator:
            self.sampler.update(initial_goals, desired_goals)

        achieved_trajectories = []
        achieved_init_states = []

        for i in range(args.episodes):
            obs = self.env_List[i].get_obs()
            init_state = obs['observation'].copy()
            
            # decide on whether to use goal generator
            if args.goal_generator:
                # generate goal by HGG or GoalGAN
                explore_goal = self.sampler.sample(i)

                # replace goal given by the environment
                self.env_List[i].goal = explore_goal.copy()
            
            # initialization for interaction with the environment
            obs = self.env_List[i].get_obs()
            current = Trajectory(obs)
            trajectory = [obs['achieved_goal'].copy()]

            for timestep in range(args.timesteps):
                action = agent.step(obs, explore=True)
                self.dynamic_buffer.add(obs['observation'].copy(), 'st')
                obss = obs.copy()
                obs, reward, done, info = self.env_List[i].step(action)
                self.dynamic_buffer.add(action.copy(), 'at')
                self.dynamic_buffer.add(obs['observation'].copy(), 'stpo')

                trajectory.append(obs['achieved_goal'].copy())
                    
                if buffer.steps_counter >= args.warmup:
                    for _ in range(self.args.training_freq):
                        batch_real = buffer.sample_batch(batch_size=12, sample_for_mb=False)
                        batch_fake = buffer_fake.sample_batch(batch_size=244)
                        batch_new = {'obs': batch_real['obs']+batch_fake['obs'],
                                     'obs_next': batch_real['obs_next']+batch_fake['obs_next'],
                                     'acts': batch_real['acts']+batch_fake['acts'],
                                     'rews': batch_real['rews']+batch_fake['rews']}
                        info = agent.train(batch_new)
                        args.logger.add_dict(info)
                agent.target_update()

                if timestep == args.timesteps-1:
                    done = True
                current.store_step(action, obs, reward, done)
                if done:
                    break

            # dynamic model training
            if args.fgi or args.model_based_training:
                # calculate delta state (st+1 minus st), which is a trick
                if self.dynamic_buffer.dynamic_buffer_number <= 1000000:
                # if len(self.st)<=20000:
                    _st = np.array(self.dynamic_buffer.data['st'][:self.dynamic_buffer.dynamic_buffer_number].copy())
                    _at = np.array(self.dynamic_buffer.data['at'][:self.dynamic_buffer.dynamic_buffer_number].copy())
                    _stpo = np.array(self.dynamic_buffer.data['stpo'][:self.dynamic_buffer.dynamic_buffer_number].copy())
                    target = _stpo - _st
                    inputs = np.concatenate([_st, _at], axis=1)
                    outputs = np.array(target)
                else:
                    _st = []
                    _at = []
                    _stpo = []
                    target = []
                    inds = np.random.randint(0, self.dynamic_buffer.dynamic_buffer_number, size=1000000)
                    for x in range(1000000):
                        _st.append(self.dynamic_buffer.data['st'][inds[x]].copy())
                        _at.append(self.dynamic_buffer.data['at'][inds[x]].copy())
                        target.append((self.dynamic_buffer.data['stpo'][inds[x]]-self.dynamic_buffer.data['st'][inds[x]]).copy())
                    _st = np.array(_st)
                    _at = np.array(_at)
                    target = np.array(target)
                    inputs = np.concatenate([_st, _at], axis=1)
                    outputs = np.array(target)
                if len(self.model_loss) > 0 and self.model_loss[-1] < 0.03:
                    los = env_model.train(inputs=inputs, targets=outputs, holdout_ratio=0.2, batch_size=256, max_epochs=10)
                else:    
                    los = env_model.train(inputs=inputs, targets=outputs, holdout_ratio=0.2, batch_size=256, max_epochs=None)
                del(_st)
                del(_at)
                del(_stpo)
                del(inputs)
                del(outputs)
                self.model_loss.append(los['val_loss'])

            # update buffer and normalizer
            achieved_trajectories.append(np.array(trajectory))
            achieved_init_states.append(init_state)
            buffer.store_trajectory(current)  

            if buffer.steps_counter > args.warmup:
                agent.normalizer_update(buffer.sample_batch()) 
            
            # generate fake data
            if buffer.steps_counter > args.warmup - 1 and args.model_based_training:
                print('extending...')
                extend_length = self.args.extend_length
                self.extend_traj(extend_length=extend_length, env_model=env_model, buffer=buffer, agent=agent, buffer_fake=buffer_fake)
                print('extend over.')

        # update achieved_trajectories for HGG sampler
        selection_trajectory_idx = {}

        for i in range(self.args.episodes):
            if goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1]) > 0.01:
                selection_trajectory_idx[i] = True
        for idx in selection_trajectory_idx.keys():
            self.achieved_trajectory_pool.insert(
                achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())
