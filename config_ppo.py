#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The config class containing all the parameters needed
zhuoxu@berkeley.edu
"""
import os


class Config_PPO:

    def __init__(self,
                 log_name,
                 s_dim=8,
                 a_dim=2,
                 vehicle_random=False,
                 vehicle_seed=0,
                 vehicle_err=0.1,
                 side_force=0,
                 track='sine_curve',
                 story_index=0,
                 n_iter=100,
                 max_path_length=1000,
                 min_timesteps_per_batch=10000,
                 gamma=0.99,
                 a_lr=1e-4,
                 c_lr=1e-4,
                 a_update_steps=100,
                 c_update_steps=100,
                 epsilon=0.2,
                 save_iter=10
                 ):

        self.log_name = log_name
        self.model_path = './' + log_name + '/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.vehicle_random = vehicle_random
        self.vehicle_seed = vehicle_seed
        self.vehicle_err = vehicle_err
        self.side_force = side_force
        self.track = track
        self.story_index = story_index

        self.n_iter = n_iter
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch
        self.gamma = gamma
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.a_update_steps = a_update_steps
        self.c_update_steps = c_update_steps
        self.epsilon = epsilon
        self.save_iter = save_iter

    def write_config_file(self):
        f = open(self.model_path + '/config_file.txt', 'w')
        f.write('model_path = %s\n\n' % self.model_path)
        f.write('parameters for the policy network\n')
        f.write('s_dim = %i\n' % self.s_dim)
        f.write('a_dim = %i\n\n' % self.a_dim)
        f.write('parameters for the environment\n')
        f.write('vehicle_random = %r\n' % self.vehicle_random)
        f.write('vehicle_seed = %i\n' % self.vehicle_seed)
        f.write('vehicle_errorBound = %f\n' % self.vehicle_err)
        f.write('track_data = %s\n' % self.track)
        f.write('story_index = %s\n\n' % self.story_index)
        f.write('parameters for the training process\n')
        f.write('n_iter = %i\n' % self.n_iter)
        f.write('max_path_length = %i\n' % self.max_path_length)
        f.write('min_timesteps_per_batch = %i\n' % self.min_timesteps_per_batch)
        f.write('gamma = %f\n' % self.gamma)
        f.write('a_lr = %f\n' % self.a_lr)
        f.write('c_lr = %f\n' % self.c_lr)
        f.write('a_update_steps = %i\n' % self.a_update_steps)
        f.write('c_update_steps = %i\n' % self.c_update_steps)
        f.write('eposilon for PPO = %f\n' % self.epsilon)
        f.write('save_iter = %i' % self.save_iter)
        f.close()
