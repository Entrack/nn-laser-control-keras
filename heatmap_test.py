import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
# from rl.policy import BoltzmannQPolicy
from rl.policy import *
from rl.memory import SequentialMemory

from heatmap_env import HeatmapEnv as selected_environment

# Make reward for going as far from the initial point
# Feed in z and N previous steps for NN to understand prev. point


ENV_NAME = 'heatmap'

env = selected_environment()
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))

# model.add(Dense(8))
# model.add(Activation('relu'))
# model.add(Dense(8))
# model.add(Activation('relu'))
# model.add(Dense(8))
# model.add(Activation('relu'))

# # 100q on 28k
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(8))
# model.add(Activation('relu'))

# # 100q on 12k, 20k, 16k, 16k, 14k, 19k
# # Valid on 25k, 25k
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(48))
# model.add(Activation('relu'))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))

# 100q on 16k, 14k, 15k, 13k, 19k, 13k, 19k, 19k, 14k
# Valid on 24k, 23k
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

nb_steps = 25000#23000#
nb_max_episode_steps_06_05_20_49 = 150
nb_max_episode_steps_06_06_16_24 = 100
nb_max_episode_steps_06_06_16_07 = 200
nb_max_episode_steps = nb_max_episode_steps_06_06_16_24

env.opt_reward = nb_max_episode_steps * 2

memory = SequentialMemory(limit=nb_steps, window_length=1)

# 0.1 : 4k, 0.25 : 4k, 0.5 : 7k-inf
policy_06_14_16_20 = BoltzmannGumbelQPolicy(C = 20.0) # 20, 50
# more stable
# 0.1 : 4k, 0.25 : 4-5k, 0.5 : 10k-inf
policy_06_13_19_00 = BoltzmannQPolicy(tau = 1.0) # 0.5
policy_06_14_16_15 = MaxBoltzmannQPolicy(eps = 0.1)
policy = policy_06_14_16_20

target_model_update_06_05_20_49 = 1e-2
target_model_update_06_05_22_18 = 1e-1
target_model_update_06_13_19_07 = 1e-3
target_model_update = target_model_update_06_05_20_49
bactch_size_06_05_22_18 = 32
bactch_size_07_05_16_07 = 64
batch_size = bactch_size_06_05_22_18
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, 
	# better
	enable_dueling_network=True, dueling_type='avg',
	# enable_double_dqn=True,
	target_model_update=target_model_update, policy=policy,
	batch_size = 32)

# Dueling is 0.8 on 20_49 hp
# Double is 0.2 on 20_49 hp
# Dueling and Double is 0.1 on 20_49 hp

lr_06_05_20_49 = 1e-3
lr_06_05_22_18 = 1e-2
lr_06_13_19_07 = 5e-4
lr = lr_06_05_20_49
dqn.compile(Adam(lr=lr), metrics=['mae'])

# dqn.load_weights('dqn_heatmap__weights.h5f')

dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=2,
 nb_max_episode_steps=nb_max_episode_steps)

dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

dqn.test(env, nb_episodes = 30, visualize=True, nb_max_episode_steps=nb_max_episode_steps)