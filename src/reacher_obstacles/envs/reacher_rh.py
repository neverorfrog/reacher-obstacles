__credits__ = ["Kallinteris-Andreas"]

import math,sys,time

from queue import Queue
from typing import Dict, Union
from inspect import getsourcefile
from os.path import abspath,dirname

import numpy as np
import cv2

import gymnasium as gym

from gymnasium import utils, Wrapper
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class RewardHeuristic(Wrapper):

    def __init__(self, env, reward_shaping = False, **kwargs):
        super().__init__(env)
        self.bins = 9
        self.gmap = np.zeros((self.bins, self.bins))
        self.last_potential = 0
        self.reward_shaping = reward_shaping
        self.abs_gamma = 0.9

    def reset(self, *, seed=None, options=None):
        obs,info = super().reset(seed=seed)

        # compute goal map

        # init
        self.gmap = np.zeros((self.bins, self.bins)) - 1  # reset to -1

        # obstacles
        for jo in range(self.env.unwrapped.nobstacles):
            obsr,obsc = self._to_r_c(self.unwrapped.obstacles[jo])
            self.gmap[obsr,obsc] = 2 * self.bins

        # zero position 
        zr,zc = self._to_r_c(np.array([0,0]))
        self.gmap[zr,zc] = 2 * self.bins


        if self.env.unwrapped.uobstacle:  # TODO make it parametric
            uobstmap = np.array( \
                    [[ 0 , 0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0,  0 ] ,
                     [ 0 , 0 , 18 , 18 , 18 ,  0 ,  0 ,  0,  0 ] ,
                     [ 0 , 0 , 18 ,  0 , 18 ,  0 ,  0 ,  0,  0 ] ,
                     [ 0 , 0 , 18 ,  0 , 18 ,  0 ,  0 ,  0,  0 ] ,
                     [ 0 , 0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0,  0 ] ,
                     [ 0 , 0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0,  0 ] ,
                     [ 0 , 0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0,  0 ] ,
                     [ 0 , 0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0,  0 ] ,
                     [ 0 , 0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0,  0 ] ])

            self.gmap = self.gmap + uobstmap


        tr,tc = self._to_r_c(self.env.unwrapped.target)

        # check that target cell is not occupied by obstacles
        assert self.gmap[tr,tc]==-1, "ERROR - reset model - target cell occupied by an obstacle!"

        self.gmap[tr,tc] = 0  # goal cell to 0

        # fill goal mapp with l1 distance to target
        q = Queue(maxsize=self.bins**2)
            
        q.put((tr,tc))
        while not q.empty():
            (r,c) = q.get()
            v = self.gmap[r,c]
            if r>0 and self.gmap[r-1,c]==-1:
                self.gmap[r-1,c] = v+1
                q.put((r-1,c))
            if r<self.bins-1 and self.gmap[r+1,c]==-1:
                self.gmap[r+1,c] = v+1
                q.put((r+1,c))
            if c>0 and self.gmap[r,c-1]==-1:
                self.gmap[r,c-1] = v+1
                q.put((r,c-1))
            if c<self.bins-1 and self.gmap[r,c+1]==-1:
                self.gmap[r,c+1] = v+1
                q.put((r,c+1))
            
        
        self.last_potential = 1.0 * (1.0 * self.abs_gamma**(self.bins*2) - 1.0)   # reset to min value

        return obs, info



    def step(self, action):
    
        observation, reward, term, trunc, info = super().step(action)
        
        reward -= info["reward_dist"]    # remove reward based on dist to target
        reward -= info["reward_ctrl"]    # remove reward based on control action

        rh, dvec = self.reward_heuristic(action)
        info['reward_heuristic'] = rh
        reward += -1 + rh  # -1 offset

        if dvec<=3:
            # close to goal, use also dist (add +1 otherwise dist will penalize the agent)
            reward += 0.5 + np.clip(info["reward_dist"],-0.5,0)
            if dvec==0:
                reward += 0.5

        return observation, reward, term, trunc, info



    # continuous x,y to discrete r,c
    def _to_r_c(self, pos):
        c = int((pos[0]+0.45)/0.1)  # TODO ok for Reacher3, make it general !!!
        r = int((0.45-pos[1])/0.1)
        return r,c


    def reward_heuristic(self, action):
    
        rh = 0
        
        ftpos = self.env.unwrapped.data.body('fingertip').xpos
        ftr,ftc = self._to_r_c(ftpos)
        
        dvec = self.gmap[ftr, ftc]

        if not self.reward_shaping:
            rh = 1.0 * (1.0 * self.abs_gamma**dvec - 1.0)   # max value 0
        else:
            potential = 1.0 * (1.0 * self.abs_gamma**dvec - 1.0)
            rh = self.abs_gamma * potential - self.last_potential
            self.last_potential = potential
        
        return rh, dvec



# env registration 

from gymnasium.envs.registration import register


def reacher_rh(**args):
    # print(args)
    render_mode = None
    if 'render_mode' in args.keys():
        render_mode=args['render_mode']
    env = gym.make(args['envid'], render_mode=render_mode)
    env = RewardHeuristic(env, **args)
    return env


def env_register(idreg, max_episode_steps=50):
    v = idreg.split('_')
    envid = v[0]+"_"+v[1]
    rs = False
    if v[2]=='rsV':
        rs = True
    register(id=idreg,
        entry_point="reacher_obstacles.envs.reacher_rh:reacher_rh",
        max_episode_steps=max_episode_steps,
        kwargs =  { 'envid': envid,
                    'reward_shaping': rs,
                     } )


rew_list = [ 'rhV', 'rsV' ]    
    
for conf in ["FT", "FTO1", "FTO1b", "FTO2", "FTO2b", "FTO2c", "FTU", "FTO3", "FTO3b"]:
  for rew in rew_list:
    env_register(f"Reacher-v6_{conf}_{rew}")
    env_register(f"Reacher3-v6_{conf}_{rew}")

for conf in ["FTUO1", "FTUO1b", "FTUO2" , "FTUO2b"]:
  for rew in rew_list:
    env_register(f"Reacher-v6_{conf}_{rew}", max_episode_steps=100)
    env_register(f"Reacher3-v6_{conf}_{rew}", max_episode_steps=100)

for conf in ["FTRO1", "FTRO2", "FTRO3" ]:
  for rew in rew_list:
    env_register(f"Reacher-v6_{conf}_{rew}")
    env_register(f"Reacher3-v6_{conf}_{rew}")

for conf in ["FT4RO1", "FT4RO2", "FT4RO3" ]:
  for rew in rew_list:
    env_register(f"Reacher-v6_{conf}_{rew}")
    env_register(f"Reacher3-v6_{conf}_{rew}")

for conf in ["RTO1", "RTO1b", "RTO2", "RTO2b", "RTO2c", "RTU", "RTUO1", "RTUO1b", "RTUO2" ]:
  for rew in rew_list:
    env_register(f"Reacher-v6_{conf}_{rew}")
    env_register(f"Reacher3-v6_{conf}_{rew}")

for conf in ["RTRO1", "RTRO2", "RTRO3" ]:
  for rew in rew_list:
    env_register(f"Reacher-v6_{conf}_{rew}")
    env_register(f"Reacher3-v6_{conf}_{rew}")




