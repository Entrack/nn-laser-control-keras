from gym import spaces
from gym import Env
from gym.envs.classic_control import rendering

import numpy as np
import time
import math
import datetime
import serial

if_virtual = True

import sys

'''
To use this script,
fill config.txt with the following values (without numbers in brackets):
(0) Measurement Lib path
(1) Serial Laser Control Lib path
(2) Signal Processing Lib path
(3) File Manager Lib path
(4) First laser diode serial port
(5) Second laser diode serial port
(6) Path to the folder for saving experimental data
(7) Temperature measuring device serial port
(8) Temperature measuring device baudrate
(9) Pre-made measurements folder path // required if you run FileVirtualDevice

'''

# LOADING CONFIG
def load_config():
	with open('config.cfg') as config:
		return [line.strip() for line in config.readlines()]

try:
	config = load_config()
	measurement_lib_path = config[0]
	serial_laser_control_lib_path = config[1]
	signal_processing_lib_path = config[2]
	file_manager_lib_path = config[3]
	first_laser_diode_serial_port = config[4]
	second_laser_diode_serial_port = config[5]
	experimental_data_saving_folder = config[6]
	temperature_serial_port = config[7]
	temperature_serial_baudrate = config[8]
	if if_virtual:
		premade_measurement_folder_path = config[9]
except:
	raise Exception("Unable to read data from config.cfg" + " \n" +
		"See the script's head to see how fill in config file")


# Path to Measurement Lib
sys.path.insert(0, measurement_lib_path)
# Path to Serial Laser Control Lib
sys.path.insert(0, serial_laser_control_lib_path)
# Path to Signal Processing Lib
sys.path.insert(0, signal_processing_lib_path)
# Path to File Manager Lib
sys.path.insert(0, file_manager_lib_path)

if if_virtual:
	from devices.FileVirtualDevice import FileVirtualDevice
else:
	from devices.ACFDevice import ACFDevice
	#from devices.RFDevice import RFDevice
	# from devices.OSDevice import OSDevice

if if_virtual:
	from lazer_serial_controllers.EmptyVirtualController import EmptyVirtualController
else:
	from lazer_serial_controllers.LazerSerialControllerSingleLD import LazerSerialControllerSingleLD

from signal_processors.ACFSignalProcessor import ACFSignalProcessor
from signal_processors.RFSignalProcessor import RFSignalProcessor
from signal_processors.OSSignalProcessor import OSSignalProcessor

from file_managers.SignalFileManager import SignalFileManager


WARNING = "\033[93m"
ENDC = '\033[0m'

class HeatmapEnv(Env):
	def __init__(self):
		self.if_print_log = False
		self.if_minimize = True
		self.good_values = {
		'ACF_fwhm_min' : 20,
		'ACF_fwhm_max' : 42,
		'ACF_coh_min' : 0,
		'ACF_coh_max' : 0.2,
		'RF_con_min' : 0,
		'RF_con_max' : 0,
		'OS_pow_min' : 0,
		'OS_pow_max' : 0
		}
		self.values_constraints = {
		'ACF_fwhm_min' : 0,
		'ACF_fwhm_max' : 160,
		'ACF_coh_min' : 0,
		'ACF_coh_max' : 1.0,
		'RF_con_min' : 0,
		'RF_con_max' : 1,
		'OS_pow_min' : 0,
		'OS_pow_max' : 1000
		}
		self.active_values = ['ACF_fwhm', 'ACF_coh']
		self.step_width = 0.1
		self.num_steps_per_episode = 100
		
		self.num_steps_to_keep = 5#8
		self.num_steps_to_check = 5#8
		self.num_z_to_keep = 3#1

		self.stay_penalty = -2.2#-2.0#-1.0
		self.step_penalty = -6.0#-5.7#-2.0
		self.edge_penalty = -10.0#-7.0
		self.path_non_consistency_penalty = -7.0 / self.num_steps_to_check
		# self.in_opt_step_penalty = -4.0
		self.opt_reward = 10.0

		self.step_time = 0.5
		self.is_scanning_current_poll_time = 0.05
		self.if_save_pics = False
		self.if_save_csv = False
		self.t_low = 10.0
		self.t_high = 50.0

		self.constraints = {
		'I1_min' : 1.0,
		'I1_max' : 3.0,
		'I2_min' : 6.0,
		'I2_max' : 8.0
		}		
		self.values= {
		'I1' : None,
		'I2' : None
		}

		if if_virtual:
			self.acf_dev = FileVirtualDevice('ACF', self.constraints, self.values, 
			premade_measurement_folder_path, file_type = 'txt', delimiter = '\t')
			# self.rf_dev = FileVirtualDevice('RF', self.constraints, self.values, 
			# premade_measurement_folder_path, file_type = 'csv', delimiter = ',')
			# self.os_dev = FileVirtualDevice('OS', self.constraints, self.values, 
			# premade_measurement_folder_path, file_type = 'csv', delimiter = ',')
		else:
			self.acf_dev = ACFDevice()
			# self.rf_dev = RFDevice()
			# self.os_dev = OSDevice()

		self.low = np.array([self.constraints['I1_min'], self.constraints['I2_min']])
		self.high = np.array([self.constraints['I1_max'], self.constraints['I2_max']])

		self.window_center = np.array([7.0, 4.0])
		self.window_radius = 1.0
		self.window_low = self.window_center - self.window_radius
		self.window_high = self.window_center + self.window_radius
		self.z_low = 0.0
		self.z_high = 160.0

		# previous actions
		self.low = np.concatenate(( self.low, np.zeros(self.num_steps_to_keep * 5) ))
		self.high = np.concatenate(( self.high, np.ones(self.num_steps_to_keep * 5) ))

		# previous z
		self.low = np.concatenate(( self.low, np.zeros(self.num_z_to_keep * len(self.active_values)) ))
		self.high = np.concatenate(( self.high, np.ones(self.num_z_to_keep * len(self.active_values)) ))

		if if_virtual:
			pass
		else:
			# shift (temperature)
			self.low = np.concatenate(( self.low, np.zeros(1) ))
			self.high = np.concatenate(( self.high, np.ones(1) ))

		# # window coords
		# self.low = np.concatenate(( self.low, np.zeros(4) ))
		# self.high = np.concatenate(( self.high, np.ones(4) ))

		self.action_space = spaces.Discrete(5)
		self.observation_space = spaces.Box(low=self.low, high=self.high)

		self.allowed_x = np.arange(self.low[0], self.high[0] + self.step_width, self.step_width)
		self.allowed_y = np.arange(self.low[1], self.high[1] + self.step_width, self.step_width)
		self.window_allowed_x = np.arange(self.window_low[0], self.window_high[0] + self.step_width, self.step_width)
		self.window_allowed_y = np.arange(self.window_low[1], self.window_high[1] + self.step_width, self.step_width)
		self.space_length = np.array([self.high[0] - self.low[0], self.high[1] - self.low[1]])
		self.window_space_length = np.array([self.window_high[0] - self.window_low[0], self.window_high[1] - self.window_low[1]])

		self.viewer = None
		self.steps_before_rendering = 0


		if if_virtual:
			self.controllers = {
			'I1' : EmptyVirtualController(),
			'I2' : EmptyVirtualController()
			}
		else:
			self.controllers = {
			'I1' : LazerSerialControllerSingleLD(first_laser_diode_serial_port),
			'I2' : LazerSerialControllerSingleLD(second_laser_diode_serial_port)
			}

		self.acf_sp = ACFSignalProcessor()
		# self.rf_sp = RFSignalProcessor()
		# self.os_sp = OSSignalProcessor()

		self.file_manager = SignalFileManager(experimental_data_saving_folder + '/csv/', 'nn_steps')

		if if_virtual:
			pass
		else:
			self.t_serial = serial.Serial(temperature_serial_port)
			self.t_serial.baudrate = temperature_serial_baudrate

		# self.values_getters = {
		# 'ACF_fwhm' : self.acf_sp.get_acf_env_fwhm,
		# 'RF_con' : None,
		# 'OS_pow' : None
		# }

		self.values['I1'] = 0.0
		self.values['I2'] = 0.0
		self.measurements= []
		self.total_steps = 0
		self.episode_actions = []
		self.last_steps = []
		self.last_values = []
		self.shift_value = 0.0

		timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
		log_filepath = "logs/log_" + str(timestamp) + ".txt"
		self.log_file = open(log_filepath,"w+")

		print(self.__class__.__name__, 'inited!')

	def step(self, action):
		self.total_steps += 1
		self.episode_actions.append(action)

		self.last_steps.append(action)
		while len(self.last_steps) > self.num_steps_to_keep:
			self.last_steps.pop(0)

		if_done = False
		reward = 0.0
		penalty = 0.0

		if action == 0:
			if not self.values['I1'] - self.step_width < self.window_allowed_x[0]:
				self.values['I1'] -= self.step_width
				self.set_current('I1', self.values['I1'])
			else:
				penalty += self.edge_penalty
		if action == 1:
			if not self.values['I1'] + self.step_width > self.window_allowed_x[-1]:
				self.values['I1'] += self.step_width
				self.set_current('I1', self.values['I1'])
			else:
				penalty += self.edge_penalty
		if action == 2:
			if not self.values['I2'] - self.step_width < self.window_allowed_y[0]:
				self.values['I2'] -= self.step_width
				self.set_current('I2', self.values['I2'])
			else:
				penalty += self.edge_penalty
		if action == 3:
			if not self.values['I2'] + self.step_width > self.window_allowed_y[-1]:
				self.values['I2'] += self.step_width
				self.set_current('I2', self.values['I2'])
			else:
				penalty += self.edge_penalty
		
		if not action == 4:
			reward += self.step_penalty
		else:
			reward += self.stay_penalty

		if not if_virtual:
			self.shift_value = self.get_shift()
		try:
			while True:
				try:
					self.measurements = self.get_values()
					if not self.measurements is None:
						break
					else:
						time.sleep(0.1)
				except KeyboardInterrupt:
					break
		except KeyboardInterrupt:
			pass

		normalized_values = []
		for idx, value_name in enumerate(self.active_values):
			normalized_values.append((self.measurements[idx] - self.values_constraints[value_name + '_min']) 
				/ (self.values_constraints[value_name + '_max'] - self.values_constraints[value_name + '_min'])) 
		self.last_values += normalized_values

		while len(self.last_values) > self.num_z_to_keep * len(self.active_values):
			self.last_values.pop(0)

		if 0 in self.last_steps and 1 in self.last_steps[-self.num_steps_to_check:]:
			reward += self.path_non_consistency_penalty
		if 2 in self.last_steps and 3 in self.last_steps[-self.num_steps_to_check:]:
			reward += self.path_non_consistency_penalty

		num_good_values = 0
		for idx, value_name in enumerate(self.active_values):
			if self.measurements[idx] <= self.good_values[value_name + '_max']:
				if self.measurements[idx] >= self.good_values[value_name + '_min']:
					num_good_values += 1
		if num_good_values == len(self.active_values):
			reward += self.opt_reward
			# if not action == 4:
			# 	reward += self.in_opt_step_penalty

		log_array = []
		log_array.append(('reward', "%.2f" % reward))
		log_array.append(('x', "%.1f" % self.values['I1']))
		log_array.append(('y', "%.1f" % self.values['I2']))
		if not if_virtual:
			log_array.append(('shift', "%.4f" % self.shift_value))
		for idx, value_name in enumerate(self.active_values):
			log_array.append((value_name, "%.4f" % self.measurements[idx]))

		file_log_str = ""
		for idx, record in enumerate(log_array):
			file_log_str += record[1]
			if not idx == len(log_array) - 1:
				 file_log_str += ','
		self.log_file.write(file_log_str + '\n')

		if self.if_print_log:
			names = ""
			values = ""
			for record in log_array:
				names += record[0] + '\t'
				values += record[1] + '\t'
				for i in range(len(record[0]) // 8):
					values += '\t'
			print(names)
			print(values)

		return self._get_obs(), reward, if_done, {}

	def get_shift(self):
		shift = 0
		# self.t_serial.write('a'.encode())
		# print(self.t_serial.readline())
		# # self.send('a')
		temperature = self.receive()
		try:
			shift = (float(temperature) - self.t_low) / (self.t_high - self.t_low)
		except:
			shift = self.shift_value
		return shift

	# def send(self, message):
	# 	self.t_serial.write((message).encode())

	def receive(self):
		line = self.t_serial.readline()
		return line.decode('utf-8').rstrip()

	def get_values(self):
		values = []

		try:
			acf_x, acf_y = self.acf_dev.get_acf()
			# os_x, os_y = self.os_dev.get_os()
			# rf_x, rf_y = self.rf_dev.get_rf()
		except:
			values = None


		if not values is None and self.if_save_pics:
			self.save_pic(acf_x, acf_y, 'acf')
			# self.save_pic(os_x, os_y, 'os')
			# self.save_pic(rf_x, rf_x, 'rf')
			
		if not values is None and self.if_save_csv:
			self.file_manager.save_acf(self.values['I1'], self.values['I2'], acf_x, acf_y)

		if not values is None:
			try:
				# for value_name in self.active_values:
				# 	values.append(self.values_getters[value_name()])
				# values.append(self.acf_sp.get_acf_env_fwhm(time=acf_x, acf=acf_y))
				acf_fwhm = self.acf_sp.get_acf_env_fwhm(time=acf_x, acf=acf_y)
				acf_coh = self.acf_sp.get_acf_coh(time=acf_x, acf=acf_y)
			except:
				values = None

		if not values is None:
			values.append(acf_fwhm)
			values.append(acf_coh)

		return values

	def set_current(self, current, value):
		self.controllers[current].scan_current(value, self.step_time)
		self.wait_for_scan()

	def wait_for_scan(self):
		while True:
			if not self.controllers['I1'].is_scanning_current() and not self.controllers['I2'].is_scanning_current():
				break
			time.sleep(self.is_scanning_current_poll_time)

	def reset(self):
		print("%.2f" % self.values['I1'], "%.2f" % self.values['I2'])
		self.values['I1'] = self.window_allowed_x[np.random.randint(0, self.window_allowed_x.size)]
		self.values['I2'] = self.window_allowed_y[np.random.randint(0, self.window_allowed_y.size)]
		self.set_current('I1', self.values['I1'])
		self.set_current('I2', self.values['I2'])

		self.last_steps = []
		for i in range(self.num_steps_to_keep):
			self.last_steps.append(-1)

		self.last_values = []
		for i in range(self.num_z_to_keep * len(self.active_values)):
			self.last_values.append(0)

		print(self.episode_actions[:-self.num_steps_to_check], self.episode_actions[-self.num_steps_to_check:])
		self.episode_actions = []
		return self._get_obs()

	def _get_obs(self):
		last_steps = np.array(self.last_steps)
		one_hot = np.zeros((last_steps.size, 5))
		one_hot[np.arange(last_steps.size), last_steps] = 1
		for i in np.arange(last_steps.size):
			if last_steps[i] < 0:
				one_hot[i] = np.zeros(5)
		one_hot = one_hot.flatten()
		# norm_window_x = ((np.array([self.window_low[0], self.window_high[0]]) - np.array([self.low[0], self.low[0]]))
		# 	/ self.space_length)
		# norm_window_y = ((np.array([self.window_low[1], self.window_high[1]]) - np.array([self.low[1], self.low[1]]))
		# 	/ self.space_length)
		if if_virtual:
			obs = np.concatenate(( ((np.array([self.values['I1'], self.values['I2']]) - np.array([self.low[0], self.low[1]]))
				# / self.space_length), one_hot, self.last_values, shift, norm_window_x, norm_window_y))
				/ self.space_length), one_hot, self.last_values))
		else:
			obs = np.concatenate(( ((np.array([self.values['I1'], self.values['I2']]) - np.array([self.low[0], self.low[1]]))
				# / self.space_length), one_hot, self.last_values, shift, norm_window_x, norm_window_y))
				/ self.space_length), one_hot, self.last_values, self.shift_value))

		return obs

	def close(self):
		pass

	def render(self, mode='human'):
		if self.total_steps < self.steps_before_rendering:
			return None
		screen_width = 400
		screen_height = 400

		x_threshold = 40

		world_width = x_threshold * 2
		scale = screen_width / world_width
		cartwidth = 30.0
		cartheight = 30.0

		if self.viewer is None:
			self.viewer = rendering.Viewer(screen_width, screen_height)

			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			self.cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			self.cart.add_attr(self.carttrans)
			self.viewer.add_geom(self.cart)

		if self.last_steps is None: return None

		# Now shift value outdated
		# screen_shift_x_y = ( np.array([7, 4]) -
		# 	np.array([self.low[0], self.low[1]]) + 
		# 	np.array((self.shift_x, self.shift_y))) 
		# 	/ self.space_length * np.array((screen_width, screen_height))

		# self.track = rendering.Line((0, screen_height / 2 + screen_shift_x_y[1]),
		#  (screen_width,screen_height / 2 + screen_shift_x_y[1]))
		# self.track.set_color(0,0,0)
		# self.viewer.add_onetime(self.track)

		# self.track2 = rendering.Line((screen_width / 2 + screen_shift_x_y[0], 0),
		#  (screen_width / 2 + screen_shift_x_y[0], screen_height))
		# self.track2.set_color(0,0,0)
		# self.viewer.add_onetime(self.track2)

		window_norm_x_y = ((np.array([self.values['I1'], self.values['I2']]) - np.array([self.window_low[0], self.window_low[1]]))
			/ self.window_space_length)
		self.carttrans.set_translation(window_norm_x_y[0] * screen_width, window_norm_x_y[1] * screen_height)

		self.cart.set_color(1., 0.0, 0.0)
		num_good_values = 0
		for idx, value_name in enumerate(self.active_values):
			if self.measurements[idx] <= self.good_values[value_name + '_max']:
				if self.measurements[idx] >= self.good_values[value_name + '_min']:
					num_good_values += 1

		if num_good_values == len(self.active_values):
			self.cart.set_color(0, 1., 0.0)
			
		return self.viewer.render(return_rgb_array = mode=='rgb_array')