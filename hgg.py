import tensorflow as tf
import numpy as np
import time
from config import get_config
from algo.hgg_learner import HGGLearner
from policy.replay_buffer import ReplayBuffer_Episodic, goal_based_process
from policy.DDPG import DDPG
from envs import make_env
from tester import Tester

if __name__ == '__main__':
    # get args
    args = get_config()
    
    # get an env
    env = make_env(args)
    env_test = make_env(args)
    if args.goal_based:
        args.obs_dims = list(goal_based_process(env.reset()).shape)
        args.acts_dims = [env.action_space.shape[0]]
        args.compute_reward = env.compute_reward
        args.compute_distance = env.compute_distance

    # initialize
    args.buffer = buffer = ReplayBuffer_Episodic(args)
    args.learner = learner = HGGLearner(args)
    args.agent = agent = DDPG(args=args, state_dim=args.obs_dims, action_dim=args.acts_dims, act_l2=args.act_l2, pi_lr=args.pi_lr, q_lr=args.q_lr,
                 clip_return_l=args.clip_return_l, clip_return_r=args.clip_return_r, gamma=args.gamma, target_update_param=args.polyak)
    args.tester = tester = Tester(args)

    args.logger.summary_init(agent.graph, agent.sess)
    # Progress info
    args.logger.add_item('Epoch')
    args.logger.add_item('Cycle')
    args.logger.add_item('Episodes@green')
    args.logger.add_item('Timesteps')
    args.logger.add_item('TimeCost(sec)')

    # Algorithm info
    for key in agent.train_info.keys():
        args.logger.add_item(key, 'scalar')

    # Test info
    for key in tester.info:
        args.logger.add_item(key, 'scalar')

    args.logger.summary_setup()


    for epoch in range(args.epoches):
        for cycle in range(args.cycles):
            args.logger.tabular_clear()
            args.logger.summary_clear()
            start_time = time.time()
            learner.learn(args, env, env_test, agent, buffer) # here env/env_test is not needed.
            tester.cycle_summary()

            # summary
            args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epoches))
            args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
            args.logger.add_record('Episodes', buffer.counter)
            args.logger.add_record('Timesteps', buffer.steps_counter)
            args.logger.add_record('TimeCost(sec)', time.time()-start_time)

            args.logger.tabular_show(args.tag)
            args.logger.summary_show(buffer.counter)
        tester.epoch_summary()

