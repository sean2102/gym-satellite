#!/usr/bin/env python3
"""Trains a DQN agent to control the satellite.

By default, it trains on the SatelliteDrag-v0 environment. Supply the script
with an environment name to train on a different environment.
"""

import sys
import os
import yaml
import gym
import tensorflow as tf
from stable_baselines.common.policies import CnnPolicy, FeedForwardPolicy
from stable_baselines import PPO2
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.schedules import LinearSchedule
import numpy as np
from enum import Enum
from datetime import datetime

import satellite_env.envs


def nature_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

#default cnn network 
class Nature_CNN(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(Nature_CNN, self).__init__(*args, **kwargs, cnn_extractor=nature_cnn,feature_extraction="cnn")


# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def train(cfg, run_name):
    startTime=datetime.now()

    save_freq= 100000

    model_path_and_name, run_description = get_model_path_and_name(cfg, run_name)

    # Create the vectorized environment   
    num_cpu = 4  # Number of processes to use
    env = SubprocVecEnv([make_env('SatelliteDrag-v0', i) for i in range(num_cpu)])

    #create a single environments
    #env = gym.make('SatelliteDrag-v0')
    #env.seed(cfg["seed"])

    #CustomPolicy = Nature_CNN

    #Learning rate scheduler
    #schedlr = LinearSchedule(cfg["time_steps"], final_p=cfg["lr_final_value"], initial_p=cfg["lr_init_value"])

    # log path
    checkpoint_path = os.path.join(os.getcwd(),"check_points")
    log_path = os.path.join(os.getcwd(), "logs", run_description + "_logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # Save checkpoints
    checkpoint_prefix = 'rl_model'
    checkpoint_callback = CheckpointCallback(save_freq=save_freq,save_path=checkpoint_path,
                                             name_prefix=checkpoint_prefix)


    model = PPO2(CustomPolicy, env, gamma=cfg["gamma"],
                            learning_rate=cfg["learning_rate"],
                            n_steps=cfg["n_steps"],
                            lam=cfg["lam"],
                            nminibatches=cfg["nminibatches"],
                            noptepochs=cfg["noptepochs"],
                            ent_coef=cfg["ent_coef"],
                            #vf_coef=cfg["vf_coef"],
                            #cliprange=cfg["cliprange"],
                            #cliprange_vf=cfg["cliprange_vf"],
                            verbose=2, tensorboard_log=log_path, full_tensorboard_log=False, seed=cfg["seed"])
        
    model.learn(total_timesteps=cfg["time_steps"], callback=checkpoint_callback, tb_log_name=run_name)

    model.save(model_path_and_name)
                    
    print('Training took ', datetime.now() - startTime, 'seconds')


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
    train(cfg = cfg, run_name = run_name)
