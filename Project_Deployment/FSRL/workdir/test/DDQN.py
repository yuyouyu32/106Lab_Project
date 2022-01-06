import numpy as np

import tensorflow as tf
# import tensorflow.compat.v1 as tf


# np.random.seed(1)

# tf.set_random_seed(1)
class DDeepQNetwork:

    def __init__(

            self,

            n_actions,

            n_features,

            learning_rate=0.01,

            reward_decay=0.9,

            e_greedy=0.9,

            replace_target_iter=500,

            memory_size=2000,

            batch_size=400,

            e_greedy_increment=None,

            output_graph=False,

            # prioritized=True,

    ):
        import os
        os.environ['KMP_WARNINGS'] = '0'

        self.n_actions = n_actions

        self.n_features = n_features

        self.lr = learning_rate

        self.gamma = reward_decay

        self.epsilon_max = e_greedy

        self.replace_target_iter = replace_target_iter

        self.memory_size = memory_size

        self.batch_size = batch_size

        self.epsilon_increment = e_greedy_increment

        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # self.prioritized = prioritized  # decide to use double q or not

        # total learning step

        self.learn_step_counter = 0

        # # initialize zero memory [s, a, r, s_]
        #
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]

        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):

            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:

            # $ tensorboard --logdir=logs

            tf.summary.FileWriter("logs/", self.sess.graph)



        self.sess.run(tf.global_variables_initializer())

        self.cost_his = []



    def _build_net(self):

        # ------------------ all inputs ------------------------

        # tf.compat.v1.disable_eager_execution()


        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State

        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State

        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward

        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # tf.set_random_seed(1)

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)




        # ------------------ build evaluate_net ------------------

        with tf.variable_scope('eval_net'):

            e1 = tf.layers.dense(self.s, 1000, tf.nn.relu, kernel_initializer=w_initializer,

                                 bias_initializer=b_initializer, name='e1')

            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,

                                          bias_initializer=b_initializer, name='q')



        # ------------------ build target_net ------------------

        with tf.variable_scope('target_net'):

            t1 = tf.layers.dense(self.s_, 1000, tf.nn.relu, kernel_initializer=w_initializer,

                                 bias_initializer=b_initializer, name='t1')

            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,

                                          bias_initializer=b_initializer, name='t2')



        with tf.variable_scope('q_target'):

            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )

            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):

            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)

            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )

        with tf.variable_scope('loss'):

            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf.variable_scope('train'):

            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)



    def store_transition(self, s, a, r, s_):

        if not hasattr(self, 'memory_counter'):

            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory

        index = self.memory_counter % self.memory_size

        self.memory[index, :] = transition

        self.memory_counter += 1


    def choose_action(self, observation,s_f):

        # to have batch dimension when feed into tf placeholder

        observation = observation[np.newaxis, :]
        # print(self.sess.run(self.q_eval, feed_dict={self.s: observation}))

        if np.random.uniform() < self.epsilon:

            # forward feed the observation and get q value for every actions

            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})

            actions = np.argmax(actions_value)

            while (s_f[actions] == 1):

                actions_value[0][actions] = actions_value[0][np.argmin(actions_value)] - 0.1

                actions = np.argmax(actions_value)

        else:

            actions = np.random.randint(0, self.n_actions)

            while (s_f[actions] == 1):

                actions = np.random.randint(0, self.n_actions)

        return actions

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

            # print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, cost = self.sess.run(

            [self._train_op, self.loss],

            feed_dict={

                self.s: batch_memory[:, :self.n_features],

                self.a: batch_memory[:, self.n_features],

                self.r: batch_memory[:, self.n_features + 1],

                self.s_: batch_memory[:, -self.n_features:],

            })

        self.cost_his.append(cost)

        # increasing epsilon

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1




    def plot_cost(self, name):

        import matplotlib.pyplot as plt
        plt.clf()

        plt.plot(np.arange(len(self.cost_his)), self.cost_his)

        plt.ylabel('Cost')

        plt.xlabel('training steps')

        # plt.show()
        plt.savefig(name)

    def choose_best_action(self, observation,s_f):

        # to have batch dimension when feed into tf placeholder

        observation = observation[np.newaxis, :]

        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})

        actions = np.argmax(actions_value)

        while (s_f[actions] == 1):

            actions_value[0][actions] = actions_value[0][np.argmin(actions_value)] - 0.1

            actions = np.argmax(actions_value)

        return actions