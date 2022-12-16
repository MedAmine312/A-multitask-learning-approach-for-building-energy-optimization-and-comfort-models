# CS 541
# RL reinforcement learning model

import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
import keras.backend as K
import tensorflow as tf
import random
from collections import deque

print(tf.__version__)

env = gym.make("Pendulum-v0")
sess = tf.compat.v1.Session()

learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.995
gamma = 0.95
tau = 0.125


#  Class Actor Critic


class Actor_Critic_Model:
    def __init__(self, env, sess):

        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.tau = 0.125

        # Setting up the Actor Model"

        self.memory = deque(maxlen=2000)

        # Actor Model State Input
        self.actor_state_input, self.actor_model = self.create_Actor_Model()
        _, self.target_actor_model = self.create_Actor_Model()

        self.actor_critic_grad = tf.compat.v1.placeholder(
            tf.float32, [None, self.env.action_space.shape[0]]
        )  # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(
            self.actor_model.output, actor_model_weights, -self.actor_critic_grad
        )  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.keras.optimizers.RMSprop(self.learning_rate).apply_gradients(
            grads
        )

        # Setting up the Critic Model"

        (
            self.critic_state_input,
            self.critic_action_input,
            self.critic_model,
        ) = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(
            self.critic_model.output, self.critic_action_input
        )  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.compat.v1.initialize_all_variables())

    def create_Actor_Model():

        state_input = Input(shape=env.observation_space.shape)
        h1 = Dense(24, activation="relu")(state_input)
        h2 = Dense(24, activation="relu")(h1)
        h3 = Dense(24, activation="relu")(h2)
        output = Dense(env.action_space.shape[0], activation="relu")(h3)

        actor = Model(inputs=state_input, outputs=output)
        adam = tf.optimizers.Adam(lr=0.001)

        actor.compile(loss="mse", optimizer=adam)

        return state_input, actor

    def create_Critic_Model():

        """
        Critic Model takes both as input the state environment and the action space and calculate a corresponding valuation
        We do this by a series of fully-connected layers, with a layer in the middle that merges the two before combining into the final Q-value prediction

        """

        state_input = Input(shape=env.observation_space.shape)
        state_h1 = Dense(24, activation="relu")(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=env.observation_space.shape)
        action_h1 = Dense(48)(action_input)

        critic_input = Add()([state_h2, action_h1])
        h1 = Dense(24, activation="relu")(critic_input)

        output = Dense(1, activation="relu")(h1)

        critic = Model(inputs=[state_input, action_input], outputs=output)

        adam = tf.optimizers.Adam(lr=0.001)

        critic.compile(loss="mse", optimizer=adam)

        return state_input, action_input, critic

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def train_actor(self, samples):
        for sample in samples:

            current_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(current_state)
            grads = self.sess.run(
                self.critic_grads,
                feed_dict={
                    self.critic_state_input: current_state,
                    self.critic_action_input: predicted_action,
                },
            )[0]
            self.sess.run(
                self.optimize,
                feed_dict={
                    self.actor_state_input: current_state,
                    self.actor_critic_grad: grads,
                },
            )

    def train_critic(self, samples):

        for sample in samples:
            current_state, action, reward, new_state, done = sample
            # print(current_state)
            # print(new_state)

            if not done:
                target_action = self.target_actor_model.predict(new_state)
                # print(target_action)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action]
                )[0][0]
                reward += gamma * future_reward

            self.critic_model.fit([current_state, action], reward, verbose=0)

    def train(self):
        batch_size = 32
        if len(memory) < batch_size:
            return
        rewards = []
        samples = random.sample(self.memory, batch_size)
        self.train_critic(samples)
        self.train_actor(samples)

    # ------ Target Model Updating

    def update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]

        self.target_critic_model.set_weights(actor_target_weights)

    def update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]

        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self.update_actor_target()
        self.update_critic_target()

    # ------ Model Prediction

    def act(self, current_state):

        self.epsilon *= self.epsilon_decay

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return self.actor_model.predict(current_state)


# %%


def main():
    sess = tf.compat.v1.Session()
    K.set_session(sess)
    env = gym.make("Pendulum-v0")
    actor_critic = Actor_Critic_Model(env, sess)

    num_trials = 10000
    trial_len = 500

    cur_state = env.reset()
    action = env.action_space.sample()

    while True:
        # env.render()
        cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
        action = actor_critic.act(cur_state)
        action = action.reshape((1, env.action_space.shape[0]))

        new_state, reward, done, _ = env.step(action)
        new_state = new_state.reshape((1, env.observation_space.shape[0]))

        actor_critic.remember(cur_state, action, reward, new_state, done)
        actor_critic.train()

        cur_state = new_state


if __name__ == "__main__":
    main()

# %% [markdown]
# # Creating the Models

# %%
def create_Actor_Model():

    state_input = Input(shape=env.observation_space.shape)
    h1 = Dense(24, activation="relu")(state_input)
    h2 = Dense(24, activation="relu")(h1)
    h3 = Dense(24, activation="relu")(h2)
    output = Dense(env.action_space.shape[0], activation="relu")(h3)

    actor = Model(inputs=state_input, outputs=output)
    adam = tf.optimizers.Adam(lr=0.001)

    actor.compile(loss="mse", optimizer=adam)

    return state_input, actor


# %%
def create_Critic_Model():

    """
    Critic Model takes both as input the state environment and the action space and calculate a corresponding valuation
    We do this by a series of fully-connected layers, with a layer in the middle that merges the two before combining into the final Q-value prediction

    """

    state_input = Input(shape=env.observation_space.shape)
    state_h1 = Dense(24, activation="relu")(state_input)
    state_h2 = Dense(48)(state_h1)

    action_input = Input(shape=env.observation_space.shape)
    action_h1 = Dense(48)(action_input)

    critic_input = Add()([state_h2, action_h1])
    h1 = Dense(24, activation="relu")(critic_input)

    output = Dense(1, activation="relu")(h1)

    critic = Model(inputs=[state_input, action_input], outputs=output)

    adam = tf.optimizers.Adam(lr=0.001)

    critic.compile(loss="mse", optimizer=adam)

    return state_input, action_input, critic


# %% [markdown]
# # Calling the Models

# %%
tf.compat.v1.disable_eager_execution()


# %%
memory = deque(maxlen=2000)
actor_state_input, actor_model = create_Actor_Model()
_, target_actor_model = create_Actor_Model()


actor_critic_grad = tf.compat.v1.placeholder(
    tf.float32, [None, env.action_space.shape[0]]
)  # where we will feed de/dC (from critic)

actor_model_weights = actor_model.trainable_weights
actor_grads = tf.gradients(
    actor_model.output, actor_model_weights, -actor_critic_grad
)  # dC/dA (from actor)

grads = zip(actor_grads, actor_model_weights)

optimize = tf.keras.optimizers.RMSprop(learning_rate).apply_gradients(grads)

# %%
critic_state_input, critic_action_input, critic_model = create_Critic_Model()
_, _, target_critic_model = create_Critic_Model()

critic_grads = tf.gradients(
    critic_model.output, critic_action_input
)  # where we calcaulte de/dC for feeding above

# Initialize for later gradient calculations
sess.run(tf.compat.v1.initialize_all_variables())

# %% [markdown]
# # Training the Models

# %%
def remember(cur_state, action, reward, new_state, done):
    memory.append([cur_state, action, reward, new_state, done])


# %%
def remember(cur_state, action, reward, new_state, done):
    memory.append([cur_state, action, reward, new_state, done])


def train_actor(samples):
    for sample in samples:
        current_state, action, reward, new_state, _ = sample
        predicted_action = actor_model.predict(current_state)
        grads = sess.run(
            critic_grads,
            feed_dict={
                critic_state_input: current_state,
                critic_action_input: predicted_action,
            },
        )[0]

        sess.run(
            optimize,
            feed_dict={actor_state_input: current_state, actor_critic_grad: grads},
        )


# %%
def train_critic(samples):
    for sample in samples:
        current_state, action, reward, new_state, done = sample
        print(current_state)
        print(new_state)

        if not done:

            target_action = target_actor_model.predict(new_state)
            print(target_action)
            future_reward = target_critic_model.predict([new_state, target_action])[0][
                0
            ]
            reward += gamma * future_reward

        #
        critic_model.fit([current_state, action], reward, verbose=0)


# %%
def train():
    batch_size = 32
    if len(memory) < batch_size:
        return

    rewards = []
    samples = random.sample(memory, batch_size)
    train_critic(samples)
    train_actor(samples)


# %%
# ------ Target Model Updating


def update_actor_target():
    actor_model_weights = actor_model.get_weights()
    actor_target_weights = target_critic_model.get_weights()

    for i in range(len(actor_target_weights)):
        actor_target_weights[i] = actor_model_weights[i]

    target_critic_model.set_weights(actor_target_weights)


# %%
def update_critic_target():
    critic_model_weights = critic_model.get_weights()
    critic_target_weights = target_critic_model.get_weights()
    for i in range(len(critic_target_weights)):
        critic_target_weights[i] = critic_model_weights[i]

    target_critic_model.set_weights(critic_target_weights)


# %%
def update_target():
    update_actor_target()
    update_critic_target()


# %%
# ------ Model Prediction


def act(current_state):
    global epsilon
    epsilon *= epsilon_decay

    if np.random.random() < epsilon:
        return env.action_space.sample()

    return actor_model.predict(current_state)


# %%
env = gym.make("Pendulum-v0")
sess = tf.compat.v1.Session()

learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.995
gamma = 0.95
tau = 0.125

num_trials = 10000
trial_len = 500

cur_state = env.reset()
action = env.action_space.sample()
while True:
    cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
    action = act(cur_state)
    action = action.reshape((1, env.action_space.shape[0]))

    new_state, reward, done, _ = env.step(action)
    new_state = new_state.reshape((1, env.observation_space.shape[0]))

    remember(cur_state, action, reward, new_state, done)
    train()

    cur_state = new_state

# %%
def create_Actor_Model():

    state_input = Input(shape=env.observation_space.shape)
    h1 = Dense(24, activation="relu")(state_input)
    h2 = Dense(24, activation="relu")(h1)
    output = Dense(env.action_space.shape[0], activation="softmax")(h2)

    actor = Model(inputs=[state_input], outputs=output)
    adam = tf.optimizers.Adam(lr=0.001)

    actor.compile(loss="mse", optimizer=adam)

    return state_input, actor


def create_Critic_Model():

    """
    Critic Model takes both as input the state environment and the action space and calculate a corresponding valuation
    We do this by a series of fully-connected layers, with a layer in the middle that merges the two before combining into the final Q-value prediction

    """

    state_input = Input(shape=env.observation_space.shape)
    state_h1 = Dense(24, activation="relu")(state_input)
    state_h2 = Dense(48, activation="relu")(state_h1)

    action_input = Input(shape=env.observation_space.shape)
    action_h1 = Dense(48, activation="relu")(state_input)

    critic_input = Add()([state_h2, action_h1])
    h1 = Dense(24, activation="relu")(critic_input)

    output = Dense(1, activation="relu")(h1)

    critic = Model(inputs=[state_input, action_input], outputs=output)

    adam = tf.optimizers.Adam(lr=0.001)

    critic.compile(loss="mse", optimizer=adam)

    return state_input, action_input, critic


memory = deque(maxlen=2000)
actor_state_input, actor_model = create_Actor_Model()
_, target_actor_model = create_Actor_Model()

actor_critic_grad = tf.compat.v1.placeholder(
    tf.float32, [None, env.action_space.shape[0]]
)  # where we will feed de/dC (from critic)

actor_model_weights = actor_model.trainable_weights

actor_grads = tf.gradients(
    actor_model.output, actor_model_weights, -actor_critic_grad
)  # dC/dA (from actor)
grads = zip(actor_grads, actor_model_weights)
optimize = tf.keras.optimizers.RMSprop(learning_rate).apply_gradients(grads)

# %%
critic_state_input, critic_action_input, critic_model = create_Critic_Model()
_, _, target_critic_model = create_Critic_Model()

critic_grads = tf.gradients(
    critic_model.output, critic_action_input
)  # where we calcaulte de/dC for feeding above

# Initialize for later gradient calculations
sess.run(tf.compat.v1.initialize_all_variables())


def remember(cur_state, action, reward, new_state, done):
    memory.append([cur_state, action, reward, new_state, done])


def train_actor(samples):
    for sample in samples:
        current_state, action, reward, new_state, _ = sample
        predicted_action = actor_model.predict(current_state)
        grads = sess.run(
            critic_grads,
            feed_dict={
                critic_state_input: current_state,
                critic_action_input: predicted_action,
            },
        )[0]

        sess.run(
            optimize,
            feed_dict={actor_state_input: current_state, actor_critic_grad: grads},
        )


def train_critic(samples):
    for sample in samples:
        current_state, action, reward, new_state, done = sample
        if not done:
            target_action = target_actor_model.predict(new_state)
            future_reward = target_critic_model.predict([new_state, target_action])[0]
            reward += gamma * future_reward

        #
        critic_model.fit([current_state, action], reward, verbose=0)


def train():
    batch_size = 32
    if len(memory) < batch_size:
        return

    rewards = []
    samples = random.sample(memory, batch_size)
    train_critic(samples)
    train_actor(samples)


# ------ Target Model Updating


def update_actor_target():
    actor_model_weights = actor_model.get_weights()
    actor_target_weights = target_critic_model.get_weights()

    for i in range(len(actor_target_weights)):
        actor_target_weights[i] = actor_model_weights[i]
        target_critic_model.set_weights(actor_target_weights)


def update_critic_target():
    critic_model_weights = critic_model.get_weights()
    critic_target_weights = target_critic_model.get_weights()
    for i in range(len(critic_target_weights)):
        critic_target_weights[i] = critic_model_weights[i]

    target_critic_model.set_weights(critic_target_weights)


def update_target():
    update_actor_target()
    update_critic_target()


# ------ Model Prediction


def act(current_state):
    global epsilon
    epsilon *= epsilon_decay

    if np.random.random() < epsilon:
        return env.action_space.sample()

    return actor_model.predict(current_state)


env = gym.make("Pendulum-v0")
sess = tf.compat.v1.Session()

learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.995
gamma = 0.95
tau = 0.125

num_trials = 10000
trial_len = 500

cur_state = env.reset()
action = env.action_space.sample()
while True:
    cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
    action = act(cur_state)
    action = action.reshape((1, env.action_space.shape[0]))

    new_state, reward, done, _ = env.step(action)
    new_state = new_state.reshape((1, env.observation_space.shape[0]))

    remember(cur_state, action, reward, new_state, done)
    train()

    cur_state = new_state
