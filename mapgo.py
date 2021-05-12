import tensorflow as tf
import numpy as np
import time
from config import get_config
from algo.hgg_learner import HGGLearner
from policy.replay_buffer import ReplayBuffer_Episodic, goal_based_process
from policy.DDPG import DDPG
from envs import make_env
from tester import Tester
from dynamic.mbpo_model.constructor import construct_model
import pandas as pd
import random

def set_global_seeds(i):
    """
    This function is an impletmentation in OpenAI baseselines.
    """
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)

if __name__ == '__main__':
    # get args
    args = get_config()
    set_global_seeds(args.seed)
    
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
    args.buffer.data_type = 'Real'
    args.buffer_fake = buffer_fake = ReplayBuffer_Episodic(args)
    args.buffer_fake.data_type = 'Fake'
    args.buffer_fake.buffer_size = 10000*3*4
    args.learner = learner = HGGLearner(args)
    args.agent = agent = DDPG(args=args, state_dim=args.obs_dims, action_dim=args.acts_dims, act_l2=args.act_l2, pi_lr=args.pi_lr, q_lr=args.q_lr,
                 clip_return_l=args.clip_return_l, clip_return_r=args.clip_return_r, gamma=args.gamma, target_update_param=args.polyak)
    args.tester = tester = Tester(args)

    args.logger.summary_init(agent.graph, agent.sess)

    # get fake env model
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    env_model_sess = tf.Session(config=cfg)
    args.env_model = env_model = construct_model(obs_dim=args.env_params['env_model_obs_dim'], act_dim=args.env_params['env_model_act_dim'], rew_dim=0, hidden_dim=args.env_hidden, num_networks=args.env_num_networks, num_elites=args.env_elites, session=env_model_sess)
    
    env_model.reset()
    env_model_sess.run(tf.global_variables_initializer())
    agent.init_network()

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
            learner.learn(args, env, env_test, agent, buffer, buffer_fake, env_model=env_model, fake=False, test=False)
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
