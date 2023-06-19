import io
import os
import pdb
import time
import json
#import orjson as json
import base64
import argparse
import threading
import importlib
from abc import ABC
from queue import Queue
from tdw.remote_build_launcher import RemoteBuildLauncher
from magnebot import Magnebot

import misc.utils as utils
from misc.utils import MongoEncoder
from environments.environment import Environment


class Room(ABC):
    """Room encapsulates one TDW room and all the agents inside the room.

    Arguments:
      room_id -- a unique identifier for each room
      cfg -- configuration of the experiment, loaded in RoomQueue
    """
    def __init__(self, room_id, cfg, **kwargs):
        self.room_id = room_id
        cfg['build_port'] = RemoteBuildLauncher.find_free_port()
        self.cfg = cfg        
        self.env = Environment(self.cfg)
        self.is_running = False
        self.agents = {}
        self.action_queue = Queue()
        self.agent_img_obs = {}
        self.all_obs = []
        self.all_actions = []
        self.setup_agents()
        self.nticks = 0
        self.last_save_tick = -1
        self.thread = None
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.room_collection = self.mongo_db[f'room_{room_id}']
        
    def save_room_data_once(self):
        data = {
            'obs': self.all_obs,
            'actions': self.all_actions,
            'step': self.nticks}
        data = json.loads(json.dumps(
            data,
            cls=MongoEncoder))        
        self.room_collection.insert_one(data)
        self.all_obs = []
        self.all_actions = []
        
    def setup_agents(self):
        self.obs = self.env.reset()
        self.enqueue_img_obs()
        self.free_agent_ids = self.env.async_agent_ids
        self.reset_env = False
        for agent_id, info in self.env.agents.items():
            init_func = info['init_func']
            is_async = self.cfg['agent_cfgs'][agent_id]['is_async']
            action_space = self.cfg['agent_cfgs'][agent_id]['action_space']
            kwargs = self.cfg['agent_cfgs'][agent_id]['kwargs']
            agent = init_func(agent_id,
                              info['avatar'],
                              is_async,
                              action_space,
                              **kwargs)
            self.agents[agent_id] = agent
            
    def get_sync_actions(self):
        """Get actions for all synchronous agents
        """
        self.actions = {}
        self.agent_cmds = {}
        for _, agent in self.agents.items():
            if not agent.is_async:
                action, cmd = agent.act(self.obs)
                self.actions[agent.avatar_id] = action
                self.agent_cmds[agent.avatar_id] = cmd
                
    def enqueue_actions(self, agent_id, action):
        """Add an async agent's action to self.action_queue
        """
        self.action_queue.put({agent_id : action})    
        
    def _convert_pil_img(self, pil_img, im_type):
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG')
        # used for HTTP streaming which only supports one user per room
        if im_type == 'bytes':
            im = buf.getvalue()
        # used for socketio
        elif im_type == 'base64':
            im = base64.b64encode(buf.getvalue())
        return im
    
    def enqueue_img_obs(self):
        """Add images of observation at current tick to the dictionary
        """
        img_obs = self.obs['imag']
        for agent_id, pil_img in img_obs.items():
            im = self._convert_pil_img(pil_img, 'bytes')
            self.agent_img_obs[agent_id] = im

    def add_agent_data_to_obs(self):
        """Agent data is returned from its own _get_summary(). This will be added
        to the self.obs
        """
        for agent_id, agent in self.agents.items():
            avsb = agent._get_summary()
            # special case for Magnebot bc its inconsistency with other TDW avatars
            # this condition might also be needed for humanoid
            if 'avsb' not in self.obs.keys():
                self.obs['avsb'] = {}
                if isinstance(agent.avatar, Magnebot):
                    avsb['dynamic'].pop('images')
            self.obs['avsb'][agent_id] = avsb
            
    def _tick(self):        
        if self.reset_env:
            self.obs = self.env.reset()
            self.enqueue_img_obs()
            self.reset_env = False            
        self.get_sync_actions()
        
        while not self.action_queue.empty():
            (agent_id, action), = self.action_queue.get().items()
            action, cmd = self.agents[agent_id].act(action)
            self.actions[agent_id] = action
            self.agent_cmds[agent_id] = cmd
            
        self.obs = self.env.step(self.agent_cmds)
        self.enqueue_img_obs()
        self.add_agent_data_to_obs()
        self.all_actions.append(self.actions)            
        self.all_obs.append(self.obs)
        self.save_room_data_once()
        self.nticks += 1        
        
    def run(self):
        print(f"\n>>>>>>>>>> Starting up room {self.room_id} <<<<<<<<<<")        
        while self.is_running:
            self._tick()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
            
    def kill(self):
        print(f"All users have left.. killing room {self.room_id}")
        self.is_running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None

            
if __name__ == '__main__':
    from config import dummy_config_sync
    room = Room(1, dummy_config_sync)
    