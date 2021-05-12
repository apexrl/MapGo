
import os
import ctypes
import numpy as np
import copy
from envs import make_env
from envs.utils import goal_distance
from policy.replay_buffer import goal_concat

def c_double(value):
	return ctypes.c_double(value)


def c_int(value):
	return ctypes.c_int(value)


def gcc_complie(c_path, so_path=None):
	assert c_path[-2:] == '.c'
	if so_path is None:
		so_path = c_path[:-2]+'.so'
	else:
		assert so_path[-3:] == '.so'
	os.system('gcc -o '+so_path+' -shared -fPIC '+c_path+' -O2')
	return so_path


def gcc_load_lib(lib_path):
	if lib_path[-2:] == '.c':
		lib_path = gcc_complie(lib_path)
	else:
		assert so_path[-3:] == '.so'
	return ctypes.cdll.LoadLibrary(lib_path)

class MatchSampler:
	def __init__(self, args, achieved_trajectory_pool):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
		self.delta = self.env.distance_threshold

		self.length = args.episodes
		init_goal = self.env.reset()['achieved_goal'].copy()
		self.pool = np.tile(init_goal[np.newaxis, :], [
		                    self.length, 1])+np.random.normal(0, self.delta, size=(self.length, self.dim))
		self.init_state = self.env.reset()['observation'].copy()

		self.match_lib = gcc_load_lib('utils/cost_flow.c')
		self.achieved_trajectory_pool = achieved_trajectory_pool

		# estimating diameter
		self.max_dis = 0
		for i in range(1000):
			obs = self.env.reset()
			dis = goal_distance(obs['achieved_goal'], obs['desired_goal'])
			if dis > self.max_dis:
				self.max_dis = dis

	def add_noise(self, pre_goal, noise_std=None):
		goal = pre_goal.copy()
		dim = 2 if self.args.env[:5] == 'Fetch' else self.dim
		if noise_std is None:
			noise_std = self.delta
		goal[:dim] += np.random.normal(0, noise_std, size=dim)
		return goal.copy()

	def sample(self, idx):
		if self.args.env[:5] == 'Fetch':
			return self.add_noise(self.pool[idx])
		else:
			return self.pool[idx].copy()

	def find(self, goal):
		res = np.sqrt(np.sum(np.square(self.pool-goal), axis=1))
		idx = np.argmin(res)
		if test_pool:
			self.args.logger.add_record('Distance/sampler', res[idx])
		return self.pool[idx].copy()

	def update(self, initial_goals, desired_goals):
		if self.achieved_trajectory_pool.counter == 0:
			self.pool = copy.deepcopy(desired_goals)
			return

		achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
		candidate_goals = []
		candidate_edges = []
		candidate_id = []

		agent = self.args.agent
		achieved_value = []
		for i in range(len(achieved_pool)):
			obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j])
                            for j in range(achieved_pool[i].shape[0])]
			feed_dict = {
				agent.state_t_input: obs
			}
			value = agent.sess.run(agent.q_pi, feed_dict)[:, 0]
			value = np.clip(value, -1.0/(1.0-self.args.gamma), 0)
			achieved_value.append(value.copy())

		n = 0
		graph_id = {'achieved': [], 'desired': []}
		for i in range(len(achieved_pool)):
			n += 1
			graph_id['achieved'].append(n)
		for i in range(len(desired_goals)):
			n += 1
			graph_id['desired'].append(n)
		n += 1
		self.match_lib.clear(n)

		for i in range(len(achieved_pool)):
			self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
		for i in range(len(achieved_pool)):
			for j in range(len(desired_goals)):
				res = np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]), axis=1)) - \
                                    achieved_value[i]/(self.args.hgg_L /
                                                       self.max_dis/(1-self.args.gamma))
				match_dis = np.min(
					res)+goal_distance(achieved_pool[i][0], initial_goals[j])*self.args.hgg_c
				match_idx = np.argmin(res)

				edge = self.match_lib.add(
					graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				candidate_goals.append(achieved_pool[i][match_idx])
				candidate_edges.append(edge)
				candidate_id.append(j)
		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0, n)
		assert match_count == self.length

		explore_goals = [0]*self.length
		for i in range(len(candidate_goals)):
			if self.match_lib.check_match(candidate_edges[i]) == 1:
				explore_goals[candidate_id[i]] = candidate_goals[i].copy()
		assert len(explore_goals) == self.length
		self.pool = np.array(explore_goals)
