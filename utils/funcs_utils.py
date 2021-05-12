import math
import numpy as np


def quaternion_to_euler_angle(array):
    # from "Energy-Based Hindsight Experience Prioritization"
    w = array[0]
    x = array[1]
    y = array[2]
    z = array[3]
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.atan2(t3, t4)
    result = np.array([X, Y, Z])
    return result

def _get_logprob(x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))
        
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds

def goal_distance(goal_a, goal_b):
    return np.linalg.norm(goal_a - goal_b, ord=2)

def compute_reward(achieved, goal, distance_threshold):
    dis = goal_distance(achieved, goal)
    return -1.0 if dis>distance_threshold else 0.0

def compute_distance(achieved, goal):
    return np.sqrt(np.sum(np.square(achieved-goal)))

### important
def step_fake(env_model, obs, action, dims, distance_threshold, args=None):
    state_t = obs['observation']
    model = env_model

    ii = np.reshape(np.concatenate([np.array(state_t),np.array(action)], axis=-1), [1, dims])

    ensemble_model_means, ensemble_model_vars = model.predict(inputs=ii, factored=True)
    ensemble_model_means[:,:,:] += state_t
    ensemble_model_stds = np.sqrt(ensemble_model_vars)
    ensemble_samples = ensemble_model_means + \
        np.random.normal(
            size=ensemble_model_means.shape) * ensemble_model_stds       
    num_models, batch_size, _ = ensemble_model_means.shape
    model_inds = model.random_inds(batch_size)
    batch_inds = np.arange(0, batch_size)
    samples = ensemble_samples[model_inds, batch_inds]
    model_means = ensemble_model_means[model_inds, batch_inds]
    model_stds = ensemble_model_stds[model_inds, batch_inds]
    log_prob, dev = _get_logprob(samples, ensemble_model_means, ensemble_model_vars)

    rewards, next_obs = samples[:, :1], samples[:, :]

    ####### format #########
    # return obs, reward, done, info
    obs_return = {}
    obs_return['observation'] = next_obs[0]
    obs_return['desired_goal'] = obs['desired_goal']
    # here for fetchpush-v1
    # obs_return['achieved_goal'] = next_obs[0][3: 6]
    # here for mountaincar
    # obs_return['achieved_goal'] = next_obs[0][0: 1]
    # here for world
    # obs_return['achieved_goal'] = next_obs[0][0: 2]
    obs_return['achieved_goal'] = next_obs[0][args.env_params['start_in_obs']:args.env_params['end_in_obs']]

    res = compute_reward(obs_return['achieved_goal'], obs_return['desired_goal'], distance_threshold)
    info = {}
    info['distance'] = goal_distance(obs_return['achieved_goal'], obs_return['desired_goal'])
    # info_return = {}
    # info_return['Distance'] = compute_distance(obs_return['achieved_goal'], obs_return['desired_goal'])
    # if info_return['Distance']<= self.args.distance_threshold:
    #     info_return['Success'] = 1
    # else:
    #     info_return['Success'] = 0

    return obs_return, res, False, info

def step_fake_batch(env_model, obs, action, dims, distance_threshold, batch, args=None):
    state_t = obs
    model = env_model

    ii = np.reshape(np.concatenate([np.array(state_t),np.array(action)], axis=-1), [batch, dims])

    ensemble_model_means, ensemble_model_vars = model.predict(inputs=ii, factored=True)

    ensemble_model_means[:,:,:] += state_t
    ensemble_model_stds = np.sqrt(ensemble_model_vars)
    ensemble_samples = ensemble_model_means + \
        np.random.normal(
            size=ensemble_model_means.shape) * ensemble_model_stds       
    num_models, batch_size, _ = ensemble_model_means.shape
    model_inds = model.random_inds(batch_size)
    batch_inds = np.arange(0, batch_size)
    samples = ensemble_samples[model_inds, batch_inds]
    model_means = ensemble_model_means[model_inds, batch_inds]
    model_stds = ensemble_model_stds[model_inds, batch_inds]
    log_prob, dev = _get_logprob(samples, ensemble_model_means, ensemble_model_vars)

    rewards, next_obs = samples[:, :1], samples[:, :]

    # next_obs = np.clip(next_obs, a_min=np.array([0,0]),a_max=np.array([20,20]))

    return next_obs

def step_fake_batch_for_world(env_model, obs, action, dims, distance_threshold, batch, args):
    # print(obs)
    # print(obs.shape)
    # print(action)
    # print(action.shape)
    return np.clip(obs+action,a_min=np.array([0,0]),a_max=np.array([20,20]))

