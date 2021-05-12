import argparse
import numpy as np
from algo import create_learner, learner_collection
from envs import make_env, clip_return_range, Robotics_envs_id
from utils.os_utils import get_arg_parser, get_logger, str2bool
import time
import os

def get_arg_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def get_config():
    parser = get_arg_parser()
    parser.add_argument('--seed', help='random seed', type=int, default=0)

    ######## hardware configure
    parser.add_argument('--gpu', help='which gpu exp assigns on', type=str, default="0")

    ######## here for extension modules
    parser.add_argument('--sr', help='if use Sibling Rivalry or not', type=str2bool, default=False)
    parser.add_argument('--goalgan', help='if use goalgan or not', type=str2bool, default=False)
    parser.add_argument('--fgi', help='relabel the goal with foresight goal inference', type=str2bool, default=False)
    parser.add_argument('--foresight_length', help='foresight length', type=int, default=10)
    parser.add_argument('--goal_generator', help='if use goal generator', type=str2bool, default=False)
    parser.add_argument('--model_based_training', help='if MB training', type=str2bool, default=False)
    parser.add_argument('--training_freq', help='training times', type=int, default=10)
    parser.add_argument('--extend_length', help='extend length', type=int, default=3)
    parser.add_argument('--her_before_fgi', help='her before fgi', type=bool, default=True)
    parser.add_argument('--test_last_step', help='judge success whether use the last step', type=bool, default=False)
    ######## here for env model configs
    parser.add_argument('--fake', help='use env model', type=str2bool, default=False)
    parser.add_argument('--env_num_networks', help='num networks', type=int, default=6)
    parser.add_argument('--env_elites', help='elites', type=int, default=3)
    parser.add_argument('--env_hidden', help='env hidden', type=int, default=200)
    parser.add_argument('--distance_threshold', help='distance_threshold', type=float, default=0.05)
    ########
    parser.add_argument(
        '--tag', help='terminal tag in logger', type=str, default='')
    parser.add_argument('--alg', help='backend algorithm',
                        type=str, default='ddpg', choices=['ddpg', 'ddpg2'])
    parser.add_argument('--learn', help='type of training method',
                        type=str, default='hgg', choices=learner_collection.keys())

    parser.add_argument('--env', help='gym env id', type=str,
                        default='FetchReach-v1') # here removed choice in envs.
    args, _ = parser.parse_known_args()
    if args.env == 'HandReach-v0':
        parser.add_argument('--goal', help='method of goal generation',
                            type=str, default='reach', choices=['vanilla', 'reach'])
    else:
        parser.add_argument('--goal', help='method of goal generation', type=str,
                            default='interval', choices=['vanilla', 'fixobj', 'interval'])
        if args.env[:5] == 'Fetch':
            parser.add_argument(
                '--init_offset', help='initial offset in fetch environments', type=np.float32, default=1.0)
        elif args.env[:4] == 'Hand':
            parser.add_argument(
                '--init_rotation', help='initial rotation in hand environments', type=np.float32, default=0.25)

    parser.add_argument('--gamma', help='discount factor',
                        type=np.float32, default=0.98)
    parser.add_argument(
        '--clip_return', help='whether to clip return value', type=str2bool, default=True)
    parser.add_argument(
        '--eps_act', help='percentage of epsilon greedy explorarion', type=np.float32, default=0.3)
    parser.add_argument(
        '--std_act', help='standard deviation of uncorrelated gaussian explorarion', type=np.float32, default=0.2)

    parser.add_argument(
        '--pi_lr', help='learning rate of policy network', type=np.float32, default=1e-3)
    parser.add_argument(
        '--q_lr', help='learning rate of value network', type=np.float32, default=1e-3)
    parser.add_argument(
        '--act_l2', help='quadratic penalty on actions', type=np.float32, default=1.0)
    parser.add_argument(
        '--polyak', help='interpolation factor in polyak averaging for DDPG', type=np.float32, default=0.95)

    parser.add_argument('--epoches', help='number of epoches',
                        type=np.int32, default=20)
    parser.add_argument(
        '--cycles', help='number of cycles per epoch', type=np.int32, default=15)
    parser.add_argument(
        '--episodes', help='number of episodes per cycle', type=np.int32, default=50)
    parser.add_argument('--timesteps', help='number of timesteps per episode',
                        type=np.int32, default=(50 if args.env[:5] == 'Fetch' else 100))
    parser.add_argument(
        '--train_batches', help='number of batches to train per episode', type=np.int32, default=20)

    parser.add_argument(
        '--buffer_size', help='number of episodes in replay buffer', type=np.int32, default=10000)
    parser.add_argument('--buffer_type', help='type of replay buffer / whether to use Energy-Based Prioritization',
                        type=str, default='normal', choices=['normal', 'energy'])
    parser.add_argument(
        '--batch_size', help='size of sample batch', type=np.int32, default=256)
    parser.add_argument(
        '--warmup', help='number of timesteps for buffer warmup', type=np.int32, default=10000)
    parser.add_argument('--her', help='type of hindsight experience replay',
                        type=str, default='future', choices=['none', 'final', 'future'])
    parser.add_argument(
        '--her_ratio', help='ratio of hindsight experience replay', type=np.float32, default=0.8)
    parser.add_argument('--pool_rule', help='rule of collecting achieved states',
                        type=str, default='full', choices=['full', 'final'])

    parser.add_argument(
        '--hgg_c', help='weight of initial distribution in flow learner', type=np.float32, default=3.0)
    parser.add_argument('--hgg_L', help='Lipschitz constant',
                        type=np.float32, default=5.0)
    parser.add_argument(
        '--hgg_pool_size', help='size of achieved trajectories pool', type=np.int32, default=1000)

    parser.add_argument('--save_acc', help='save successful rate',
                        type=str2bool, default=True)

    args = parser.parse_args()
    
    # gpu visible setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    args.goal_based = True
    args.clip_return_l, args.clip_return_r = clip_return_range(args)

    logger_name = args.alg+'-'+args.env+'-'+args.learn
    if args.tag != '':
        logger_name = args.tag+'-'+logger_name
    args.logger = get_logger(logger_name)

    for key, value in args.__dict__.items():
        if key != 'logger':
            args.logger.info('{}: {}'.format(key, value))
    
    # predefine the corresponding dims in different envs.
    env_alt = {
        'Mountaincar-v0': {
            'start_in_obs': 0,
            'end_in_obs': 1,
            'desire_dim': 1,
            'step_fake_param': 3,
            'env_model_obs_dim': 2,
            'env_model_act_dim': 1
        },
        'FetchPush-v1': {
            'start_in_obs': 3,
            'end_in_obs': 6,
            'desire_dim': 3,
            'step_fake_param': 29,
            'env_model_obs_dim': 25,
            'env_model_act_dim': 4
        },
        'World-v0': {
            'start_in_obs': 0,
            'end_in_obs': 2,
            'desire_dim': 2,
            'step_fake_param': 4,
            'env_model_obs_dim': 2,
            'env_model_act_dim': 2
        },
        'FetchReach-v1': {
            'start_in_obs': 0,
            'end_in_obs': 3,
            'desire_dim': 3,
            'step_fake_param': 14,
            'env_model_obs_dim': 10,
            'env_model_act_dim': 4
        },
        'Pendulum-v0': {
            'start_in_obs': 1,
            'end_in_obs': 3,
            'desire_dim': 2,
            'step_fake_param': 4,
            'env_model_obs_dim': 3,
            'env_model_act_dim': 1
        },
        'AntLocomotion-v0': {
            'start_in_obs': 0,
            'end_in_obs': 2,
            'desire_dim': 2,
            'step_fake_param': 37,
            'env_model_obs_dim': 29,
            'env_model_act_dim': 8
        },
        'AntLocomotionDiverse-v0': {
            'start_in_obs': 0,
            'end_in_obs': 2,
            'desire_dim': 2,
            'step_fake_param': 37,
            'env_model_obs_dim': 29,
            'env_model_act_dim': 8
        },
        'HalfCheetahGoal-v0': {
            'start_in_obs': 8,
            'end_in_obs': 9,
            'desire_dim': 1,
            'step_fake_param': 23,
            'env_model_obs_dim': 17,
            'env_model_act_dim': 6
        },
        'Reacher-v0': {
            'start_in_obs': 9,
            'end_in_obs': 11,
            'desire_dim': 2,
            'step_fake_param': 14,
            'env_model_obs_dim': 11,
            'env_model_act_dim': 3
        }
    }

    args.env_params = env_alt[args.env]

    args.model_loss_log_name = args.tag + time.strftime('-(%Y-%m-%d-%H:%M:%S)')

    return args

