import gym
import envs.fetch as fetch_env
import envs.hand as hand_env
import envs.mountain_car as mountain_car
import envs.world as world
import envs.pendulum as pendulum
import envs.ant_locomotion as ant_locomotion
import envs.ant_locomotion_diverse as ant_locomotion_diverse
import envs.half_cheetah as half_cheetah
import envs.reacher as reacher
from .utils import goal_distance, goal_distance_obs

Robotics_envs_id = [
	'FetchReach-v1',
	'FetchPush-v1',
	'FetchPushDisentangled-v1',
	'FetchSlide-v1',
	'FetchPickAndPlace-v1',
	'HandManipulateBlock-v0',
	'HandManipulateEgg-v0',
	'HandManipulatePen-v0',
	'HandReach-v0'
]

def make_env(args):
	# assert args.env in Robotics_envs_id
	if args.env == 'Mountaincar-v0':
		return mountain_car.make_env(args)
	if args.env == 'World-v0':
		return world.make_env(args)
	if args.env == 'Pendulum-v0':
		return pendulum.make_env(args)
	if args.env == 'AntLocomotion-v0':
		return ant_locomotion.make_env(args)
	if args.env == 'AntLocomotionDiverse-v0':
		return ant_locomotion_diverse.make_env(args)
	if args.env == 'HalfCheetahGoal-v0':
		return half_cheetah.make_env(args)
	if args.env == 'Reacher-v0':
		return reacher.make_env(args)
	if args.env[:5]=='Fetch':
		return fetch_env.make_env(args)
	else: # Hand envs
		return hand_env.make_env(args)

def clip_return_range(args):
	gamma_sum = 1.0/(1.0-args.gamma)
	return {
		'FetchReach-v1': (-gamma_sum, 0.0),
		'FetchPush-v1': (-gamma_sum, 0.0),
		'FetchPushDisentangled-v1': (-gamma_sum, 0.0),
		'FetchSlide-v1': (-gamma_sum, 0.0),
		'FetchPickAndPlace-v1': (-gamma_sum, 0.0),
		'HandManipulateBlock-v0': (-gamma_sum, 0.0),
		'HandManipulateEgg-v0': (-gamma_sum, 0.0),
		'HandManipulatePen-v0': (-gamma_sum, 0.0),
		'HandReach-v0': (-gamma_sum, 0.0),
		'Mountaincar-v0': (-gamma_sum, 0.0),
		'World-v0': (-gamma_sum, 0.0),
		'Pendulum-v0': (-gamma_sum, 0.0),
		'AntLocomotion-v0': (-gamma_sum, 0.0),
		'AntLocomotionDiverse-v0': (-gamma_sum, 0.0),
		'HalfCheetahGoal-v0': (-gamma_sum, 0.0),
		'Reacher-v0': (-gamma_sum, 0.0)
	}[args.env]
