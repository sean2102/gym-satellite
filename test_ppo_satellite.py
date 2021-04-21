#!/usr/bin/env python3
import sys
import os
import yaml
import gym
import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.schedules import LinearSchedule
import numpy as np
from enum import Enum
from datetime import datetime
import imageio

import satellite_env.envs

def test(cfg, intermediate_models_to_test, run_name):
	print("Starting tests")

	startTime=datetime.now()

	model_path_and_name, run_description = get_model_path_and_name(cfg, run_name)

	# num_cpu = 4  # Number of processes to use
	# # Create the vectorized environment
	# env = SubprocVecEnv([make_env(cfg["env_name"], i,mapped_spaces=mapped_buildings, max_number_of_steps=cfg["max_number_of_steps"],exposure_distance=cfg["exposure_distance"],
	#                     random_starting_point=(cfg["starting_point"] == 'random')) for i in range(num_cpu)])
	env = gym.make('SatelliteDrag-v0')
	env.seed(cfg["seed"])


	models_to_test = intermediate_models_to_test + [model_path_and_name]
	res = {}
	steps = []
	success_ratio = []
	av_steps_in_success = []
	av_wall_heating = []

	for m in models_to_test:
		if isinstance (m,int):
		    model_path = os.path.join(os.gcwd(), "check_points", "rl_model_%d_steps.zip"%m)
		else:
		    model_path = model_path_and_name

		model = PPO2.load(model_path_and_name)

		obs = env.reset()
		images = []
		video_path = os.path.join(os.getcwd(),"videos")

		if isinstance(m, int):
		    filename = run_description + '_' + run_name + '_test_%dsteps.gif'%(m)
		else:
		    filename = run_description + '_' + run_name + '_test.gif'

		# obs = env.reset()
		# img = env.render(mode='rgb_array')
		# for i in range(350):
		#     images.append(img)
		#     action, _ = model.predict(obs)
		#     obs, _, _ ,_ = env.step(action)
		#     img = env.render(mode='rgb_array')

		# imageio.mimsave(os.path.join(video_path, filename), [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

		# Evaluate the agent
		episode_reward = 0
		for _ in range(3000):
		  action, _ = model.predict(obs)
		  obs, reward, done, info = env.step(action)
		  env.render()
		  episode_reward += reward
		  if done or info.get('is_success', False):
		    print("Reward:", episode_reward, "Success?", info.get('is_success', False))
		    episode_reward = 0.0
		    obs = env.reset()
		    break
		env.close()

		print ("model: ",m)

	print('Testing took ', datetime.now() - startTime, 'seconds')



def get_model_path_and_name(cfg, run_name):
    run_description = "ppo2"
    if run_name == None:
        run_name = "lr_%.6f" % cfg["learning_rate"]
    model_path_and_name = os.path.join(os.getcwd(), "trained_models", run_description + '_' + run_name + ".zip")
    return model_path_and_name, run_description



if __name__ == "__main__":
    cfg = yaml.safe_load(open("cfg_ppo.yml"))

    run_name = "1m_zooparams"

    np.set_printoptions(3)
    print(tf.Session(config=tf.ConfigProto(log_device_placement=True)))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    test(cfg=cfg, intermediate_models_to_test = [], run_name = run_name)
