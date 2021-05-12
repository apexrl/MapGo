import tensorflow as tf
import numpy as np
from utils.network_utils import get_vars, Normalizer
from policy.replay_buffer import goal_based_process
from utils.funcs_utils import step_fake_batch
from envs.utils import goal_distance

class DDPG:
    def __init__(self, args, state_dim, action_dim, act_l2, pi_lr, q_lr, clip_return_l, clip_return_r, gamma, target_update_param):
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.act_l2 = act_l2
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.clip_return_l = clip_return_l
        self.clip_return_r = clip_return_r
        self.gamma = gamma
        self.target_update_param = target_update_param

        self.build_network()
        self.train_info_pi = {
            'Pi_q_loss': self.pi_q_loss,
            'Pi_l2_loss': self.pi_l2_loss
        }

        self.train_info_q = {
            'Q_loss': self.q_loss
        }
        self.train_info = {**self.train_info_pi, **self.train_info_q}

        self.step_info = {
            'Q_average': self.q_pi
        }

    def build_network(self):
        def mlp_policy(obs_ph):
            with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                pi_dense1 = tf.layers.dense(
                    obs_ph, 256, activation=tf.nn.relu, name='pi_dense1')
                pi_dense2 = tf.layers.dense(
                    pi_dense1, 256, activation=tf.nn.relu, name='pi_dense2')
                pi_dense3 = tf.layers.dense(
                    pi_dense2, 256, activation=tf.nn.relu, name='pi_dense3')
                pi = tf.layers.dense(
                    pi_dense3, self.action_dim[0], activation=tf.nn.tanh, name='pi')
            return pi

        def mlp_value(obs_ph, acts_ph):
            state_ph = tf.concat([obs_ph, acts_ph], axis=1)
            with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                q_dense1 = tf.layers.dense(state_ph, 256, activation=tf.nn.relu, name='q_dense1')
                q_dense2 = tf.layers.dense(q_dense1, 256, activation=tf.nn.relu, name='q_dense2')
                q_dense3 = tf.layers.dense(q_dense2, 256, activation=tf.nn.relu, name='q_dense3')
                q = tf.layers.dense(q_dense3, 1, name='q')
            return q

        self.graph = tf.Graph()
        with self.graph.as_default():
            # session
            cfg = tf.ConfigProto()
            cfg.gpu_options.allow_growth = True
            self.sess = tf.Session(config=cfg)

            # placeholders
            self.state_t_input = tf.placeholder(tf.float32, [None]+self.state_dim)
            self.state_tpo_input = tf.placeholder(tf.float32, [None]+self.state_dim)
            self.action_input = tf.placeholder(tf.float32, [None]+self.action_dim)
            self.rewards_input = tf.placeholder(tf.float32, [None, 1])

            # normalizer
            with tf.variable_scope('normalizer'):
                self.state_normalizer = Normalizer(self.state_dim, self.sess)
            self.state_t = self.state_normalizer.normalize(self.state_t_input)
            self.state_tpo = self.state_normalizer.normalize(self.state_tpo_input)

            # network
            with tf.variable_scope('main'):
                with tf.variable_scope('policy'):
                    self.pi = mlp_policy(self.state_t)
                with tf.variable_scope('value'):
                    self.q = mlp_value(self.state_t, self.action_input)
                with tf.variable_scope('value', reuse=True):
                    self.q_pi = mlp_value(self.state_t, self.pi)
            
            with tf.variable_scope('target'):
                with tf.variable_scope('policy'):
                    self.pi_target = mlp_policy(self.state_tpo)
                with tf.variable_scope('value'):
                    self.q_target = mlp_value(self.state_tpo, self.pi_target)
            
            # operators
            self.pi_q_loss = -tf.reduce_mean(self.q_pi)
            self.pi_l2_loss = self.act_l2*tf.reduce_mean(tf.square(self.pi))
            self.pi_optimizer = tf.train.AdamOptimizer(self.pi_lr)
            self.pi_train_op = self.pi_optimizer.minimize(self.pi_q_loss+self.pi_l2_loss, var_list=get_vars('main/policy'))

            return_value = tf.clip_by_value(self.q_target, self.clip_return_l, self.clip_return_r)
            target = tf.stop_gradient(self.rewards_input+self.gamma*return_value)

            self.q_loss = tf.reduce_mean(tf.square(self.q-target))
            self.q_optimizer = tf.train.AdamOptimizer(self.q_lr)
            self.q_train_op = self.q_optimizer.minimize(self.q_loss, var_list=get_vars('main/value'))

            self.target_update_op = tf.group([
                v_t.assign(self.target_update_param*v_t + (1.0-self.target_update_param)*v) for v, v_t in zip(get_vars('main'), get_vars('target'))
            ])

            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
            self.target_init_op = tf.group([
                v_t.assign(v)
                for v, v_t in zip(get_vars('main'), get_vars('target'))
            ])
        self.init_network()

    def init_network(self):
        self.sess.run(self.init_op)
        self.sess.run(self.target_init_op)
    
    def step(self, obs, explore=False, test_info=False):
        if (not test_info) and (self.args.buffer.steps_counter<self.args.warmup):
            return np.random.uniform(-1, 1, size=self.args.acts_dims)
        if self.args.goal_based: obs = goal_based_process(obs)

        # eps-greedy exploration
        if explore and np.random.uniform()<=self.args.eps_act:
            return np.random.uniform(-1, 1, size=self.args.acts_dims)

        feed_dict = {
            self.state_t_input: [obs]
        }
        action, info = self.sess.run([self.pi, self.step_info], feed_dict)
        action = action[0]

        # uncorrelated gaussian explorarion
        if explore: action += np.random.normal(0, self.args.std_act, size=self.args.acts_dims)
        action = np.clip(action, -1, 1)

        if test_info: return action, info
        return action

    def fd_generate(self, batch):
        pass

    def step_batch(self, obs, batch_size=None, explore=False):
        actions = self.sess.run(self.pi, {self.state_t_input: obs})
        if explore and np.random.uniform()<=self.args.eps_act:
            return np.random.uniform(-1, 1, size=(batch_size, self.args.acts_dims[0]))
        if explore:
            actions += np.random.normal(0, self.args.std_act, size=(batch_size, self.args.acts_dims[0]))
        # print(np.random.normal(0, self.args.std_act, size=(batch_size, self.args.acts_dims[0])))
        actions = np.clip(actions, -1, 1)
        return actions

    def feed_dict(self, batch):
        return {
            self.state_t_input: batch['obs'],
            self.state_tpo_input: batch['obs_next'],
            self.action_input: batch['acts'],
            self.rewards_input: batch['rews']
        }

    def train(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _, _ = self.sess.run([self.train_info, self.pi_train_op, self.q_train_op], feed_dict)
        return info

    def train_pi(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _ = self.sess.run([self.train_info_pi, self.pi_train_op], feed_dict)
        return info

    def train_q(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _ = self.sess.run([self.train_info_q, self.q_train_op], feed_dict)
        return info

    def normalizer_update(self, batch):
        self.state_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))

    def target_update(self):
        self.sess.run(self.target_update_op)

    def mpc_step(self, env_model, obs, args, desired_goal):
        mpc_sample = 10
        mpc_step = 5

        pure_obs_batch = np.array([obs['observation'] for _ in range(mpc_sample)])
        desired_goal_batch = np.array([obs['desired_goal'] for _ in range(mpc_sample)])
        selected = False
        original_acts = None
        for x in range(mpc_step):
            pi_input = np.concatenate([pure_obs_batch, desired_goal_batch], axis=1)
            actions = self.step_batch(pi_input, explore=True, batch_size=10)
            if selected == False:
                original_acts = actions.copy()
                selected = True
            pure_obs_batch = step_fake_batch(env_model=env_model, obs=pure_obs_batch, action=actions, dims=args.env_param['step_fake_param'], distance_threshold=args.distance_threshold, args=args, batch=10)
        min_id = -1
        min_dis = 999999999999
        for x in range(mpc_sample):
            achieved = pure_obs_batch[mpc_step-1][args.env_param['start_in_obs']:args.env_param['end_in_obs']]
            if goal_distance(desired_goal, achieved) < min_dis:
                min_dis = goal_distance(desired_goal, achieved)
                min_id = x
        return original_acts[min_id]