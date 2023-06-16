import io
import os
import pdb
import json
import time
import base64
import pickle
import pandas as pd
import numpy as np
import argparse
import threading
import importlib
from abc import ABC
from queue import Queue
from tdw.remote_build_launcher import RemoteBuildLauncher
from tdw.add_ons.image_capture import ImageCapture

import misc.utils as utils
from misc.analysis_utils import save_traj_observations
from configs.dataset_generation_config import generate_config
from environments.environment import Environment


class Room(ABC):
    def __init__(self, room_id, cfg, time_limit):
        self.room_id = room_id
        cfg['build_port'] = RemoteBuildLauncher.find_free_port()
        self.cfg = cfg        
        self.env = Environment(self.cfg)
        self.is_running = False
        self.agents = {}
        self.setup_agents()
        self.nticks = 0
        self.time_limit = time_limit
        self.all_actions = []
        self.all_obs = []
                
    def setup_agents(self):
        self.obs = self.env.reset()
        self.reset_env = False
        for agent_id, info in self.env.agents.items():
            init_func = info['init_func']
            is_async = self.cfg['agent_cfgs'][agent_id]['is_async']
            action_space = self.cfg['agent_cfgs'][agent_id]['action_space']
            kwargs = self.cfg['agent_cfgs'][agent_id]['kwargs']
            agent = init_func(agent_id, info['avatar'], is_async, action_space, **kwargs)
            self.agents[agent_id] = agent
            
    def get_actions(self):
        self.actions = {}
        self.agent_cmds = {}
        for _, agent in self.agents.items():
            if not agent.is_async:
                action, cmd = agent.act(self.obs)
                self.actions[agent.avatar_id] = action
                self.agent_cmds[agent.avatar_id] = cmd
                          

    def add_agent_data_to_obs(self):
        for agent_id, agent in self.agents.items():
            if not agent.is_async:
                if 'avsb' not in self.obs.keys():
                    self.obs['avsb'] = {}
                self.obs['avsb'][agent_id] = agent._get_summary() 
        # add occupancy map to obs
        self.obs['occ_map'] = self.env.occupancy_map
            
    def _tick(self):
        self.nticks += 1
        if self.nticks%50 == 0:
            print(self.nticks)
        if self.reset_env:
            self.obs = self.env.reset()
            self.reset_env = False            
        self.get_actions()
        obs_comm = None
        obs_path = None
        obs_path_dict = None
        # self.obs = self.env.step(self.agent_cmds)
        # self.add_agent_data_to_obs()
        # if applicable, copy over information communicated globally for coordinating multi agent behavior
        if self.obs and 'comm' in self.obs.keys():
            obs_comm = self.obs['comm']
        # if self.obs and 'path' in self.obs.keys():
        #     print('Here')
        #     obs_path = self.obs['path']
        if self.obs and 'path_dict' in self.obs.keys():
            obs_path_dict = self.obs['path_dict']
        elif self.obs:
            obs_path_dict = {}
        self.obs = self.env.step(self.agent_cmds)
        self.add_agent_data_to_obs()
        if self.obs and 'path' in self.obs.keys():
            obs_path = self.obs['path']
        if obs_comm is not None:
            self.obs['comm'] = obs_comm
        if obs_path_dict is not None:
            self.obs['path_dict'] = obs_path_dict
        if obs_path is not None:
            self.obs['path_dict']['latest'] = obs_path
        # if obs_path is not None:
        #     self.obs['path'] = obs_path
        self.all_actions.append(self.actions)
        self.all_obs.append(self.obs)
      
    def start(self):
        self.is_running = True
        # create an occupancy map while ignoring the agents that move around
        self.env.occupancy_map.generate(ignore_objects=list(self.agents.keys()))
        resp = self.env.communicate([])
        suppress_rendering = False
        if suppress_rendering and 'disable_cameras' in self.cfg['img_settings'] and self.cfg['img_settings']['disable_cameras']:
            cmds = []
            for agent_id, info in self.env.agents.items():
                cmds.append({"$type": "enable_image_sensor", "enable": False, 
                             "sensor_name": "SensorContainer", "avatar_id": str(agent_id)})
            resp = self.env.communicate(cmds)
        # cmds = [{"$type": "send_images", "frequency": "once", "ids": ["fp1"]}]
        # resp = self.env.communicate(cmds)
        self.run()
        
    def run(self):        
        while self.is_running and self.nticks < self.time_limit:
            self._tick()    
        capture = ImageCapture(avatar_ids=['fp1'], path=self.env.img_settings['img_path'])
        capture.set(save=self.env.img_settings['save_img'], frequency="once") 
        self.env.add_ons.extend([capture])
        resp = self.env.communicate([])
        #save observations
        if 'save_obs' in self.cfg['img_settings'] and self.cfg['img_settings']['save_obs']:
            obs_save_dir = self.cfg['img_settings']['img_path']
            save_traj_observations(self.all_obs, self.cfg, obs_save_dir)
            actions_save_file = self.cfg['img_settings']['img_path']+'trial_actions.pkl'
            with open(actions_save_file, 'wb') as handle:
                pickle.dump(self.all_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
def save_config(config):
    save_file = config['img_settings']['img_path']+'config.pkl'
    with open(save_file, 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
if __name__ == '__main__':
    # time limit of simulation
    time_limit = 1500
    # number of trials to generate
    num_trials2gen = 1000
    for i in range(num_trials2gen):
        config = generate_config()
        # print behaviors and timestamp
        print(config['img_settings']['img_path'])
        
        room = Room(1, config, time_limit)  
        room.start()
        
        save_config(config)
        resp = room.env.communicate({"$type": "terminate"})
        del room