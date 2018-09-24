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
# import time
# for i in range(5000):
# 	x, y = env.acf_dev.get_acf()
# 	print(env.acf_sp.get_acf_env_fwhm(x, y))
# 	# print(env.acf_sp.get_fwhm_test(x, y))
# 	# print(env.get_shift())
# 	time.sleep(0.2)
# exit(0)
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

# # 100q on 16k, 14k, 15k, 13k, 19k, 13k, 19k, 19k, 14k
# # Valid on 24k, 23k
# model.add(Dense(96))
# model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(32))
# model.add(Activation('relu'))

# # 100q on 16k, 14k, 15k, 13k, 19k, 13k, 19k, 19k, 14k
# # Valid on 24k, 23k
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(96))
# model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(32))
# model.add(Activation('relu'))

# 100q on 16k, 14k, 15k, 13k, 19k, 13k, 19k, 19k, 14k
# Valid on 24k, 23k
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

nb_max_episode_steps = env.num_steps_per_episode
env.steps_before_rendering = 0#55000#45000
nb_steps = 60000#50000

# if_load = True
if_load = False
weights_name = 'dqn_weights_2018-07-02_20-03'
if_learn = True
# if_learn = False

memory = SequentialMemory(limit=nb_steps, window_length=1)

# 0.1 : 4k, 0.25 : 4k, 0.5 : 7k-inf
policy_06_14_16_20 = BoltzmannGumbelQPolicy(C = 20.0) # 20, 50
# more stable
# 0.1 : 4k, 0.25 : 4-5k, 0.5 : 10k-inf
policy_06_13_19_00 = BoltzmannQPolicy(tau = 1.0) # 0.5
policy_06_14_16_15 = MaxBoltzmannQPolicy(eps = 0.5)
policy = policy_06_14_16_20

target_model_update_06_05_20_49 = 1e-2
target_model_update_06_05_22_18 = 1e-1
target_model_update_06_13_19_07 = 1e-3
target_model_update = target_model_update_06_05_20_49
bactch_size_06_05_22_18 = 32
bactch_size_07_05_16_07 = 64
batch_size = bactch_size_06_05_22_18
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, 
	enable_dueling_network=True, dueling_type='avg',
	# enable_double_dqn=True,
	target_model_update=target_model_update, policy=policy,
	batch_size = 32)

lr_06_05_20_49 = 1e-3
lr_06_05_22_18 = 1e-2
lr_06_13_19_07 = 5e-4
lr = lr_06_05_20_49
dqn.compile(Adam(lr=lr), metrics=['mae'])

import datetime
if if_learn:
	if if_load:
		dqn.load_weights('weights/' + weights_name + '.h5f')
	try:
		dqn.fit(env, nb_steps=nb_steps, visualize=True, verbose=2,
	 		nb_max_episode_steps=nb_max_episode_steps)
	except KeyboardInterrupt:
		pass
	dqn.save_weights('weights/dqn_weights_{}.h5f'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")), overwrite=True)
else:
	dqn.load_weights('weights/' + weights_name + '.h5f')	
	env.steps_before_rendering = 0


dqn.test(env, nb_episodes = 30, visualize=True, nb_max_episode_steps=nb_max_episode_steps)