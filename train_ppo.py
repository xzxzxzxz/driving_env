#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The class used to train the policy using ppo
zhuoxu@berkeley.edu
"""

import numpy as np
import driving
from config_ppo import Config_PPO
from ppo import PPO
import time
import csv
import threading
import tensorflow as tf


def get_time_str():
    t = time.localtime()
    return "_%i_%i_%i_%i_%i_%i" % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


def get_log_name(str):
    return str + get_time_str()


class PPO_Train:
    def __init__(self, config):
        self.config = config
        self.env = driving.Driving(track_data=self.config.track,
                                   story_index=self.config.story_index,
                                   vh_random_model=self.config.vehicle_random,
                                   vh_random_seed=self.config.vehicle_seed,
                                   vh_err_bound=self.config.vehicle_err,
                                   vh_side_force=self.config.side_force)
        self.config.s_dim = self.env.getObservSpaceDim()
        self.config.a_dim = self.env.getActionSpaceDim()
        self.config.write_config_file()
        self.ppo = PPO(self.config)

    def restore(self,
                log_name='model',
                ckpt_name='model.ckpt'):
        model_path = './' + log_name + '/'
        self.ppo.saver.restore(self.ppo.sess, model_path + ckpt_name)
        print("expert model from " + model_path + ckpt_name + " restored.")

    def train(self):
        f = open(self.config.model_path + '/log_file.csv', 'w')
        log_writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(['iteration', 'average return', 'std of return',
                             'average ep_len', 'std of ep_len'])
        total_timesteps = 0
        for itr in range(self.config.n_iter):

            # Collect paths until we have enough timesteps
            timesteps_this_batch = 0
            paths = []
            while True:
                ob = self.env.reset()
                obs, acs, rewards = [], [], []
                steps = 0
                while True:
                    obs.append(ob)
                    ac = self.ppo.choose_action(ob)
                    acs.append(ac)
                    ob, rew, done = self.env.step(ac, steps)
                    rewards.append(rew)
                    steps += 1
                    if done or steps > self.config.max_path_length:
                        break
                path = {"observation": np.array(obs),
                        "reward": np.array(rewards),
                        "action": np.array(acs)}
                paths.append(path)
                timesteps_this_batch += len(path["reward"])  # i.e. += pathlength(path)
                if timesteps_this_batch > self.config.min_timesteps_per_batch:
                    break
            total_timesteps += timesteps_this_batch

            # Build arrays for observation, action for the policy gradient
            # update by concatenating across paths
            ob_no = np.concatenate([path["observation"] for path in paths])
            ac_na = np.concatenate([path["action"] for path in paths])

            # computing the q values
            q_t_list = []
            for path in paths:
                reward = path["reward"]
                path_len = reward.shape[0]
                q_t = np.zeros(reward.shape)
                for i in range(path_len):
                    t = path_len - 1 - i
                    if t == path_len - 1:
                        q_t[t] = reward[t]
                    else:
                        q_t[t] = reward[t] + self.config.gamma * q_t[t + 1]
                q_t_list.append(q_t)
            q_n = np.concatenate([q_t for q_t in q_t_list])[:, np.newaxis]

            # computing baseline
            # output from critic network is ~N(0,1), tfadv accepts ~N(0,1)
            b_n = self.ppo.get_v(ob_no)
            b_n = q_n.mean() + q_n.std() * b_n  # supposed we train b_n to have mean=0 and std=1
            adv_n = q_n - b_n

            # advantage normalization and q value normalization
            adv_n = (adv_n - adv_n.mean()) / adv_n.std()
            q_n = (q_n - q_n.mean()) / q_n.std()

            # optimizing actor and critic networks
            closs_before = self.ppo.get_closs(ob_no, q_n)
            aloss_before = self.ppo.get_aloss(ob_no, ac_na, adv_n)
            self.ppo.update(ob_no, ac_na, q_n, adv_n)
            closs_after = self.ppo.get_closs(ob_no, q_n)
            aloss_after = self.ppo.get_aloss(ob_no, ac_na, adv_n)

            returns = [path["reward"].sum() for path in paths]
            ep_length = [len(path["reward"]) for path in paths]
            print("********** Vehicle %i Iteration %i ************" % (self.config.vehicle_seed, itr))
            print("average return: ", np.mean(returns))
            print("std of return:  ", np.std(returns))
            print("average ep_len: ", np.mean(ep_length))
            print("std of ep_len:  ", np.std(ep_length))
            print("aloss_before:   ", aloss_before)
            print("aloss_after:    ", aloss_after)
            print("closs_before:   ", closs_before)
            print("closs_after:    ", closs_after)
            log_writer.writerow([itr, np.mean(returns), np.std(returns), np.mean(ep_length), np.std(ep_length)])

            if (itr + 1) % self.config.save_iter == 0:
                if (itr + 1) < self.config.n_iter:
                    save_path = self.ppo.saver.save(self.ppo.sess,
                                                    self.config.model_path + "model_" + str(itr+1) + '.ckpt')
                    print("Policy saved in file: %s" % save_path)
                    self.env.render(self.config.model_path + "video_" + str(itr+1), show=False, debugview=True)
        f.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.rand:
        log_name = get_log_name("tar_vh" + str(args.seed))
        config_ppo = Config_PPO(log_name=log_name,
                                vehicle_random=True,
                                vehicle_seed=args.seed)
    else:
        log_name = get_log_name("src_vh")
        config_ppo = Config_PPO(log_name=log_name)

    train_ppo = PPO_Train(config=config_ppo)
    train_ppo.restore()
    train_ppo.train()
