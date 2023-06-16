import pdb
from abc import ABC, abstractmethod

import json
import math
import random
import numpy as np
import os
import re
from scipy.spatial import distance

from tdw.tdw_utils import TDWUtils
from tdw.backend.encoder import Encoder
from tdw.add_ons.avatar_body import AvatarBody
from magnebot import Magnebot, Arm, ActionStatus


class Agent(ABC):
    """Abstract class for agent
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, **kwargs):
        self.avatar_id = avatar_id
        self.avatar = avatar
        self.is_async = is_async
        self.action_space = action_space
        for k, v in kwargs.items():
            self.k = v
            
    def _get_summary(self):
        # this is necessary because when saving images from magnebots, the images
        # are part of the avatar summary. Remove them to save processing time
        if isinstance(self.avatar, Magnebot):
            self.avatar.dynamic.images = {}
        return json.loads(json.dumps(self.avatar, cls=Encoder, indent=2, sort_keys=True))
        
    def create_action_to_cmds_dict(self):
        """
        create a dictionary that maps agent actions to TDW commands
        """
        pass

    """ Return the action from its own action space as well as TDW commands for the action """
    @abstractmethod
    def act(self, obs):
        pass


class PremotorAgent(Agent):
    def __init__(self, avatar_id, avatar, is_async, action_space, **kwargs): 
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "PremotorAgent"
        self.prev_action = None        
        
    def get_action_cmds(self, action):
        self.cont_action = action == self.prev_action and self.avatar.action.status == ActionStatus.ongoing
        if not self.cont_action:
            #print(f"running function for new action {action} {kwargs}")
            action_func, kwargs = self.action_to_cmds_dict[action]
            action_func(**kwargs)
            #print(f"running function for new action {action} {kwargs}")
            print(f"{self.avatar_id} running function for new action {action} {kwargs}")
        self.prev_action = action        
        return []
    
    def get_initial_position(self):
        x_pos = self.avatar.initial_position['x']
        y_pos = self.avatar.initial_position['y']
        z_pos = self.avatar.initial_position['z']
        return {"x": x_pos, "y": y_pos, "z": z_pos}
    
    def move_on_path(self, obs, self_pos, target_pos, state, arrived_offset):
        #path_found = 'path' in obs.keys() and len(obs['path']) > 0
        path_found = 'path_found_'+str(self.avatar_id) in obs['comm']
        # path_in_obs = len(obs['path_dict'][self.avatar_id]) > 0 or len(obs['path_dict']['latest']) > 0
        # path_found = 'path_found_'+str(self.avatar_id) in obs['comm'] and path_in_obs
        #if not path_found:
        # if self.avatar.action.status == ActionStatus.failed_to_move:
        #     # action = 'move_forward'
        #     # kwargs = {"distance": 0.5}
        #     # action_func = self.avatar.move_by
        #     # self.action_to_cmds_dict[action] = (action_func, kwargs)
        #     # cmds = self.get_action_cmds(action)
        #     action = "move"
        #     next_to_target = target_pos
        #     kwargs = {"target": next_to_target, "arrived_offset": arrived_offset}
        #     action_func = self.avatar.move_to
        #     if self.prev_action == action:
        #         state = [self.avatar.action.status == ActionStatus.success]

        #if not path_found or ('path' in obs and len(obs['path']) == 0):
        if not path_found:
            action = 'find_path'
            obj_keys = list(obs['tran'].keys())
            cmds = []
            # add all objects that aren't the current goal to the nav mesh
            # for key in obj_keys:
            #     obj_pos = obs['tran'][key]
            #     dist2obj = distance.euclidean(obj_pos, target_pos)
            #     # don't add object that could be the current goal - is very close to target location
            #     if dist2obj > 0.2:
            #         cmds.append({"$type": "make_nav_mesh_obstacle", "id": key, "carve_type": "all", "scale": 1, "shape": "box"})
            self_pos = self_pos.astype(float)
            target_pos = target_pos.astype(float)
            cmds.append({"$type": "send_nav_mesh_path",
                    "origin": {"x": self_pos[0], "y": self_pos[1], "z": self_pos[2]},
                    "destination": {"x": target_pos[0], "y": target_pos[1], "z": target_pos[2]},
                    "id": self.avatar_id})
            #temp
            # cmds.append({"$type": "send_is_on_nav_mesh", "position": {"x": 0.0, 
            #                                                           "y": 0.0, "z": -1.0}})
            # cmds.append({"$type": "send_is_on_nav_mesh", "position": {"x": target_pos[0], 
            #                                                           "y": target_pos[1], "z": target_pos[2]}})
            state = []
        elif len(state) == 0:
        #elif state == 'sending_nav_mesh_path':
            path_locs = obs['path_dict']['latest'].copy()
            if len(path_locs) > 0 and path_locs[0] == 'invalid':
                print('INVALID PATH',self.avatar_id)
                state = ['invalid', False]
                #state = [False]
            elif len(path_locs) > 0:
                # put path info in key for avatar id - to maintain compatiblity for multi-agent settings
                obs['path_dict'][self.avatar_id] = path_locs[1:]
                state = [False for i in range(len(path_locs[1:]))]
            else:
                action = 'noop'
                cmds = []
            if len(state) == 0:
                print(path_locs)
            
        if path_found and len(state) > 0:
            if state[0] == 'invalid':
                print('INVALID PATH2',self.avatar_id)
                action = 'move_on_invalid_path'
                print(target_pos)
                if self.prev_action == action:
                    if self.avatar.action.status == ActionStatus.success:
                        print('STATE=TRUE')
                        state = [True]
                kwargs = {"target": target_pos, "arrived_offset": arrived_offset}
                action_func = self.avatar.move_to
                self.action_to_cmds_dict[action] = (action_func, kwargs)
                cmds = self.get_action_cmds(action)
                # if self.prev_action == action:
                #     if self.avatar.action.status == ActionStatus.success:
                #         state = [True]
            elif all(state):
                action = 'finished'
                cmds = []
            else:
                #path_locs = obs['path'][1:]
                path_locs = obs['path_dict'][self.avatar_id]
                loc_num = np.min([i for i, x in enumerate(state) if not x])
                action = "move"
                target_pos = path_locs[loc_num]
                if self.prev_action == action and distance.euclidean(self_pos, target_pos) < 0.75:
                    state[loc_num] = self.avatar.action.status == ActionStatus.success
                    if state[loc_num]:
                        loc_num+=1
                if all(state):
                    action = 'finished'
                    cmds = []
                else:
                    p = TDWUtils.array_to_vector3(target_pos)
                    p["y"] = 0
                    kwargs = {"target": p, "arrived_offset": arrived_offset}
                    # # set y to 0
                    # target_pos[1] = 0.0
                    # kwargs = {"target": target_pos, "arrived_offset": arrived_offset}
                    action_func = self.avatar.move_to
                    self.action_to_cmds_dict[action] = (action_func, kwargs)
                    cmds = self.get_action_cmds(action)
            
        return action, cmds, state
    
    def pick_up_on_path(self, obs, self_pos, obj_pos, ob_key, state, pick_up_offset):
        if len(state) == 0 or not all(state):
            action, cmds, state = self.move_on_path(obs, self_pos, obj_pos, state, pick_up_offset)
        else:
            obj_held = bool(self.avatar.dynamic.held[Arm.left])
            if not obj_held:
                action = "grab"
                kwargs = {"target": ob_key, "arm": Arm.left}
                action_func = self.avatar.grasp
            else:
                action = "reset_arm"
                kwargs = {"arm": Arm.left}
                action_func = self.avatar.reset_arm
                state = 'Done'
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
                
        return action, cmds, state
    
    def pick_up(self, self_pos, obj_pos, ob_key, pick_up_offset):
        dist2target = distance.euclidean(obj_pos, self_pos)
        if dist2target > 0.65:
            action = "move"
            kwargs = {"target": obj_pos, "arrived_offset": pick_up_offset}
            action_func = self.avatar.move_to
        else:
            obj_held = bool(self.avatar.dynamic.held[Arm.left])
            if not obj_held:
                action = "grab"
                kwargs = {"target": ob_key, "arm": Arm.left}
                action_func = self.avatar.grasp
            else:
                action = "reset_arm"
                kwargs = {"arm": Arm.left}
                action_func = self.avatar.reset_arm
            
        return action, action_func, kwargs
    
    def drop_off(self, self_pos, target_pos, ob_key, drop_offset):
        dist2target = distance.euclidean(target_pos, self_pos)
        if dist2target > drop_offset:
            action = "move"
            kwargs = {"target": target_pos, "arrived_offset": drop_offset-0.3}
            action_func = self.avatar.move_to
        else:
            obj_held = bool(self.avatar.dynamic.held[Arm.left])
            if obj_held:
                action = "drop"
                kwargs = {"target": ob_key, "arm": Arm.left}
                action_func = self.avatar.drop
            else:
                action = "reset_arm"
                kwargs = {"arm": Arm.left}
                action_func = self.avatar.reset_arm
        
        return action, action_func, kwargs
    
    def drop_on_obj(self, obs, self_pos, target_pos, turn_target, reach_target, ob_key, drop_offset, state, path_state):
        if not state[0]:
            # go a little behind target before rotating to prevent collision with table
            if turn_target[2] < 0:
                target_pos[2] = target_pos[2] + 0.1
            else:
                target_pos[2] = target_pos[2] - 0.1
            action, cmds, path_state = self.move_on_path(obs, self_pos, target_pos, path_state, drop_offset)
            if all(path_state) and len(path_state) > 0:
                state[0] = True
        elif not state[1]:
            # turn to face the table at a 90 degree angle
            action = 'turn_to'
            kwargs = {"target": turn_target, "aligned_at": 2}
            action_func = self.avatar.turn_to
            if self.prev_action == action:
                state[1] = self.avatar.action.status == ActionStatus.success
            still_moving_backward = self.prev_action == 'move_backward' and self.avatar.action.status == ActionStatus.ongoing
            if self.avatar.action.status == ActionStatus.collision or still_moving_backward:
                #rotating may cause a collision
                action = 'move_backward'
                kwargs = {"distance": -0.2}
                action_func = self.avatar.move_by
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
        elif not state[2]:
            # in case you moved backwards while rotating, move forward to target again
            dist2target = np.abs(target_pos[2] - self_pos[2])
            #if dist2target > drop_offset:
            if dist2target > 0.05:
                action = "move_forward"
                # kwargs = {"target": target_pos, "arrived_offset": drop_offset}
                # action_func = self.avatar.move_to
                #kwargs = {"distance": dist2target, "arrived_at": drop_offset}
                kwargs = {"distance": dist2target, "arrived_at": 0.01}
                action_func = self.avatar.move_by
                if self.prev_action == action:
                    state[2] = self.avatar.action.status == ActionStatus.success
                # moving forward action may cause collision if holding object touches table
                # if so, move ahead to next action
                if self.avatar.action.status == ActionStatus.collision:
                    state[2] = True
                self.action_to_cmds_dict[action] = (action_func, kwargs)
                cmds = self.get_action_cmds(action)
            else:
                state[2] = True
                #action = 'noop'
                #cmds = []
                action = "raise_torso"
                y_pos = 3.0
                kwargs = {"height": y_pos}
                action_func = self.avatar.slide_torso
                if self.prev_action == action:
                    state[3] = self.avatar.action.status == ActionStatus.success
                self.action_to_cmds_dict[action] = (action_func, kwargs)
                cmds = self.get_action_cmds(action)
        elif not state[3]:
            action = "raise_torso"
            y_pos = 3.0
            kwargs = {"height": y_pos}
            action_func = self.avatar.slide_torso
            if self.prev_action == action:
                state[3] = self.avatar.action.status == ActionStatus.success
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
        elif not state[4]:
            # arm_id = self.avatar.static.joint_ids_by_name['magnet_left']
            # arm_position = self.avatar.dynamic.joints[arm_id].position
            # print(arm_position)
            # reach_target = arm_position.copy()
            # reach_target[1] = reach_target[1] + 0.0
            # reach_target[2] = reach_target[2] - 0.2
            action = 'reach'
            kwargs = {"target": reach_target, "arm": Arm.left, "absolute": True, "arrived_at": 0.125}
            action_func = self.avatar.reach_for
            if self.prev_action == action:
                state[4] = self.avatar.action.status == ActionStatus.success
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
        elif not state[5]:
            action = "drop"
            kwargs = {"target": ob_key, "arm": Arm.left}
            action_func = self.avatar.drop
            if self.prev_action == action:
                state[5] = self.avatar.action.status == ActionStatus.success
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
        elif not state[6]:
            action = "move_back"
            kwargs = {"distance": -1.0}
            action_func = self.avatar.move_by
            if self.prev_action == action:
                state[6] = self.avatar.action.status == ActionStatus.success
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
        elif not state[7]:
            action = "reset_arm"
            kwargs = {"arm": Arm.left}
            action_func = self.avatar.reset_arm
            if self.prev_action == action:
                state[7] = self.avatar.action.status == ActionStatus.success
                if state[7] == True:
                    print('ARM RESET')
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
        # else:
        #     obj_held = bool(self.avatar.dynamic.held[Arm.left])
        #     if obj_held:
        #         action = "drop"
        #         kwargs = {"target": ob_key, "arm": Arm.left}
        #         action_func = self.avatar.drop
        #     else:
        #         action = "reset_arm"
        #         kwargs = {"arm": Arm.left}
        #         action_func = self.avatar.reset_arm
        # else:
        #     action = 'noop'
        #     action_func = None
        #     kwargs = {}
        
        return action, cmds, state, path_state
    
    def drop_on_obj2(self, self_pos, target_pos, turn_target, reach_target, ob_key, drop_offset, state):
        if not state[0]:
            action = "move"
            next_to_target = target_pos
            kwargs = {"target": next_to_target, "arrived_offset": drop_offset}
            action_func = self.avatar.move_to
            if self.prev_action == action:
                state[0] = self.avatar.action.status == ActionStatus.success
        elif not state[1]:
            # turn to face the table at a 90 degree angle
            action = 'turn_to'
            kwargs = {"target": turn_target}
            action_func = self.avatar.turn_to
            if self.prev_action == action:
                state[1] = self.avatar.action.status == ActionStatus.success
        elif not state[2]:
            action = "raise_torso"
            y_pos = 3.0
            kwargs = {"height": y_pos}
            action_func = self.avatar.slide_torso
            if self.prev_action == action:
                state[2] = self.avatar.action.status == ActionStatus.success
        elif not state[3]:
            # arm_id = self.avatar.static.joint_ids_by_name['magnet_left']
            # arm_position = self.avatar.dynamic.joints[arm_id].position
            # #print(self.avatar.dynamic.joints[arm_id].position)
            # above_target = arm_position
            # above_target[1] = above_target[1] + 0.4
            # above_target[2] = above_target[2] - 0.3
            # if ActionStatus == ActionStatus.failed_to_reach:
            #    above_target[2] = above_target[2] - 0.25 
            action = 'reach'
            kwargs = {"target": reach_target, "arm": Arm.left, "absolute": True, "arrived_at": 0.125}
            action_func = self.avatar.reach_for
            if self.prev_action == action:
                state[3] = self.avatar.action.status == ActionStatus.success
        elif not state[4]:
            action = "drop"
            kwargs = {"target": ob_key, "arm": Arm.left}
            action_func = self.avatar.drop
            if self.prev_action == action:
                state[4] = self.avatar.action.status == ActionStatus.success
        elif not state[5]:
            action = "move_back"
            kwargs = {"distance": -1.0}
            action_func = self.avatar.move_by
            if self.prev_action == action:
                state[5] = self.avatar.action.status == ActionStatus.success
        elif not state[6]:
            action = "reset_arm"
            kwargs = {"arm": Arm.left}
            action_func = self.avatar.reset_arm
            if self.prev_action == action:
                state[6] = self.avatar.action.status == ActionStatus.success
        # else:
        #     obj_held = bool(self.avatar.dynamic.held[Arm.left])
        #     if obj_held:
        #         action = "drop"
        #         kwargs = {"target": ob_key, "arm": Arm.left}
        #         action_func = self.avatar.drop
        #     else:
        #         action = "reset_arm"
        #         kwargs = {"arm": Arm.left}
        #         action_func = self.avatar.reset_arm
        # else:
        #     action = 'noop'
        #     action_func = None
        #     kwargs = {}
        
        return action, action_func, kwargs, state
            
        
class StaticMagnebotAgent(PremotorAgent):
    """StaticMagnebotAgent does not move
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "StaticMagnebotAgent"
        self.counter = 0
        self.create_action_to_cmds_dict()
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):
        cmds = []
        return 'static', cmds
    
    
class RandomArmMagnebotAgent(PremotorAgent):
    """A magnebot randomly swinging its arms
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "RandomArmMagnebotAgent"
        self.create_action_to_cmds_dict()
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):
        target = {
            "x": np.random.uniform(low=-0.2, high=0.2),
            "y": 0.5,
            "z": np.random.uniform(low=-0.3, high=0.3)}
        if self.avatar.action.status in [
                ActionStatus.cannot_reach, ActionStatus.failed_to_reach]:
            if self.which_arm:
                action = "reset_arm"
                kwargs = {"arm": self.which_arm}
                self.action_to_cmds_dict[action] = (self.avatar.reset_arm, kwargs)
        else:
            action = np.random.choice(list(["reach_left", "reach_right"]))
            self.which_arm = Arm.left if action == "reach_left" else Arm.right
            kwargs = {"target": target, "arm": self.which_arm, 'absolute': False}
            self.action_to_cmds_dict[action] = (self.avatar.reach_for, kwargs)
        cmds = self.get_action_cmds(action)
        return action, cmds
    

class RandomMovingMagnebotAgent(PremotorAgent):
    """A Magnebot randomly moving in the room. This agent shouldn't be initialized too closely to the boundary of the room.
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, save_commands=False, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "RandomMovingMagnebotAgent"
        self.create_action_to_cmds_dict()
        self.save_commands = save_commands
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):
        if "occ_map" in obs:            
            if self.avatar.action.status == ActionStatus.collision:                
                position = self.get_initial_position()
            else:
                position = {
                    "x": np.random.uniform(
                        obs["occ_map"].scene_bounds.x_min + 1,
                        obs["occ_map"].scene_bounds.x_max - 1),
                    "y": 0,
                    "z": np.random.uniform(
                        obs["occ_map"].scene_bounds.z_min + 1,
                        obs["occ_map"].scene_bounds.z_max - 1)}
        else:
            position = {
                "x": self.avatar.initial_position['x'] + 0.5,
                "y": 0,                
                "z": self.avatar.initial_position['z'] - 0.5}
            
        arrived_offset = 0.3
        kwargs = {"target": position, "arrived_offset": arrived_offset}
        action = "move"
        self.action_to_cmds_dict[action] = (self.avatar.move_to, kwargs)        
        cmds = self.get_action_cmds(action)
        if self.save_commands and not self.cont_action and "comm" in obs:
            obs["comm"].append(TDWUtils.vector3_to_array(position))
            
        return action, cmds

    
class PeriodicMovingMagnebotAgent(PremotorAgent):
    """A Magnebot randomly moving in the room. This agent shouldn't be initialized too closely to the boundary of the room.
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "PeriodicMovingMagnebotAgent"
        self.create_action_to_cmds_dict()
        self.counter = 0
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):
        # when collision happens, return to its initial position
        if self.avatar.action.status == ActionStatus.collision:
            position = self.get_initial_position()            
        elif self.counter % 4 == 0:
            # center of the upper-right quadrant
            if "occ_map" in obs:
                position = {
                    "x": obs["occ_map"].scene_bounds.x_max / 2,
                    "y": 0,
                    "z": obs["occ_map"].scene_bounds.z_max / 2}
            else:
                position = {
                    "x": self.avatar.initial_position['x'],
                    "y": self.avatar.initial_position['y'],
                    "z": self.avatar.initial_position['z']}
        elif self.counter % 4 == 1:
            # center of the bottom-right quadrant
            position = {
                "x": obs["occ_map"].scene_bounds.x_max / 2,
                "y": 0,
                "z": obs["occ_map"].scene_bounds.z_min / 2}
        elif self.counter % 4 == 2:
            # center of the bottom-left quadrant
            position = {
                "x": obs["occ_map"].scene_bounds.x_min / 2,
                "y": 0,
                "z": obs["occ_map"].scene_bounds.z_min / 2}
            
        kwargs = {"target": position}
        action = "move"
        self.action_to_cmds_dict[action] = (self.avatar.move_to, kwargs)        
        cmds = self.get_action_cmds(action)
        return action, cmds
    

class PeriodicArmMagnebotAgent(PremotorAgent):
    """A Magnebot randomly moving in the room. This agent shouldn't be initialized too closely to the boundary of the room.
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "PeriodicArmMagnebotAgent"
        self.create_action_to_cmds_dict()
        self.reached = False
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):
        if not self.reached:
            action = "reach"
            self_pos = self.avatar.dynamic.transform.position
            target = {'x': self_pos[0] - 0.2, 'y': self_pos[1], 'z': self_pos[2]}
            kwargs = {"target": target, 'arm': Arm.left}                        
            self.action_to_cmds_dict[action] = (self.avatar.reach_for, kwargs)
            self.reached = self.avatar.action.status == ActionStatus.success
        else:
            action = "reset"            
            kwargs = {'arm': Arm.left}
            self.action_to_cmds_dict[action] = (self.avatar.reset_arm, kwargs)
            self.reached = not self.avatar.action.status == ActionStatus.success
        cmds = self.get_action_cmds(action)
        return action, cmds
    

class DeterminsiticTowerBuildingMagnebotAgent(PremotorAgent):
    """An agent building a tower with blocks
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "DeterminsiticTowerBuildingMagnebotAgent"
        self.create_action_to_cmds_dict()
        for  k, v in kwargs.items():
            setattr(self, k, v)
        self.all_objects = [1, 2, 3, 4, 5]
        self.stacked_objects = []
        self.holding_object = False
        self.dropoff_ready = False
        self.dropoff_complete = False
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}

    def act(self, obs):
        if self.dropoff_complete:            
            self.stacked_objects.append(self.all_objects.pop())
            print("popped!", self.all_objects)
            self.dropoff_complete = False
            self.holding_object = False
        obs_keys = obs['tran'].keys()
        ob_key = self.all_objects[-1]
        obj_pos = obs['tran'][ob_key]
        self_pos = self.avatar.dynamic.transform.position
        dist2target = distance.euclidean(obj_pos, self_pos)        
        if not self.holding_object:
            # empty hands
            if dist2target < 0.5:
                # object within distance
                action = "grasp"
                kwargs = {"target": ob_key, "arm": Arm.left}
                self.action_to_cmds_dict[action] = (self.avatar.grasp, kwargs)                
                self.holding_object = self.avatar.action.status == ActionStatus.success
                print(action, self.avatar.action.status)
            else:
                # object too far, need to move closer
                action = "move_to_object"
                kwargs = {"target": ob_key, "arrived_offset": 0.25}
                self.action_to_cmds_dict[action] = (self.avatar.move_to, kwargs)
                print(action, self.avatar.action.status)
        else:
            # holding an object
            target_position = np.array(list(self.target_location.values()))
            dist2drop = distance.euclidean(self_pos, target_position)
            if dist2drop < 0.5:                
                # at the drop spot
                #if self.dropoff_ready:
                # object is in a position ready to be dropped                    
                action = "drop"
                kwargs = {"target": ob_key, "arm": Arm.left}
                self.action_to_cmds_dict[action] = (self.avatar.drop, kwargs)
                self.dropoff_complete = self.avatar.action.status == ActionStatus.success
                print(action, self.avatar.action.status)
                #else:
                #    if len(self.stacked_objects) > 0:
                #        target = self.stacked_objects[-1]
                #    else:
                #        target = self.target_location
                #    action = "lower_object"
                #    kwargs = {"target": target,
                #              "arm": Arm.left,
                #              "orientation_mode": OrientationMode.z}
                #    self.action_to_cmds_dict[action] = (self.avatar.reach_for, kwargs)
                #    self.dropoff_ready = self.avatar.action.status == ActionStatus.success
                #    print(action, self.avatar.action.status)
            else:
                # need to move to drop spot
                action = "move_to_target"
                kwargs = {"target": self.target_location}
                self.action_to_cmds_dict[action] = (self.avatar.move_to, kwargs)
                print(action, self.avatar.action.status)
        cmds = self.get_action_cmds(action)        
        return action, cmds


class DummyMagnebotAgent(PremotorAgent):
    """A dummy magnebot example for using Magnebot agent
    as a subclass of PremotorAgent
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "DummyMagnebotAgent"
        self.counter = 0
        self.create_action_to_cmds_dict()
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):
        target = {
            "x": np.random.uniform(low=-0.2, high=0.2),
            "y": 0.5,
            "z": np.random.uniform(low=-0.5, high=0.5)
        }
        if self.counter % 60:
            action = "move_left"
            kwargs = {"target": target, "arm": Arm.left, 'absolute': False}    # Assume it's already relative position
            self.action_to_cmds_dict[action] = \
                (self.avatar.reach_for, kwargs)        
        else:
            action = "move_right"            
            kwargs = {"target": target, "arm": Arm.right}
            self.action_to_cmds_dict[action] = \
                (self.avatar.reach_for, kwargs)
        cmds = self.get_action_cmds(action)
        self.counter += 1
        return action, cmds
    

class GraspMagnebotAgent(PremotorAgent):
    """A dummy magnebot example for using Magnebot agent
    as a subclass of PremotorAgent
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "DummyMovingMagnebotAgent"
        self.counter = 0
        self.create_action_to_cmds_dict()
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):        
        distance = np.random.uniform(-1.5, 1.5)
        action = "move"
        kwargs = {"distance": distance}
        self.action_to_cmds_dict[action] = (self.avatar.move_by, kwargs)
        cmds = self.get_action_cmds(action)
        self.counter += 1
        return action, cmds
    
    
class GatheringMagnebotAgent(PremotorAgent):
    """A dummy magnebot example for using Magnebot agent
    as a subclass of PremotorAgent
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, target_obj, bring2observer, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "DummyMovingMagnebotAgent"
        self.counter = 0
        self.create_action_to_cmds_dict()
        self.target_obj = target_obj
        self.bring2observer = bring2observer
        self.start = False
        self.arm_reset = False
        self.goal_flags = [False, False]
        self.state = []
        self.path_state = []
        self.reset_pos = False
        self.other_agent_init_pos = None
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):   
        if not self.start:
            self.start = True
            obs['comm'] = ['start']
        
        obs_keys = list(obs['tran'].keys())
        ob_key = obs_keys[self.target_obj]
        obj_pos = obs['tran'][ob_key]
        obj_pos = obj_pos.astype(float)
        self_pos = self.avatar.dynamic.transform.position
        self_pos = self_pos.astype(float)
        dist2target = distance.euclidean(obj_pos, self_pos)
        
        ob_key2 = obs_keys[1]
        obj_pos2 = obs['tran'][ob_key2]
        
        if self.counter > 0:
            fp1_pos = np.array(obs['avsb']['fp1']['transform']['position'])
            if self.other_agent_init_pos is None:
                avsb_keys = list(obs['avsb'].keys())
                self.other_agent_id = [key for key in avsb_keys if key not in ['fp1', self.avatar_id]][0]
                self.other_agent_init_pos = TDWUtils.vector3_to_array(obs['avsb'][self.other_agent_id]['initial_position'])
        else:
            self.init_self_pos = self_pos
            self.obj_init_pos = obj_pos
            
        # check for collisions
        if self.avatar.action.status == ActionStatus.collision or (self.prev_action == 'move_backward_collision' and self.avatar.action.status == ActionStatus.ongoing):
            collision_event = 'agent_'+str(self.avatar_id)+' collision frame_'+str(self.counter)
            print(collision_event)
            obs['comm'].append(collision_event)
            if self.prev_action != 'move_backward_collision':
                action = 'move_backward_collision'
                kwargs = {"distance": -1.5}
                action_func = self.avatar.move_by
                self.action_to_cmds_dict[action] = (action_func, kwargs)
                cmds = self.get_action_cmds(action)
            else:
                cmds = []
                action = 'noop'
                
            self.counter += 1
            return action, cmds
        
        # if adversary takes object back to original location
        if all(self.goal_flags):
            obj_back = False
            for entry in obs['comm']:
                # if object was returned by adversary
                obj_close = distance.euclidean(self.obj_init_pos, obj_pos) < 1.0
                other_agent_pos = obs['avsb'][self.other_agent_id]['dynamic']['transform']['position']
                other_agent_back = distance.euclidean(other_agent_pos, self.other_agent_init_pos) < 1.0
                if 'return_'+str(self.target_obj) in entry and obj_close and other_agent_back:
                    obj_back = True
            if obj_back:
                print('Obj back')
                self.goal_flags = [False, False]
                self.state = []
                self.path_state = []
                self.reset_pos = False
                
        if not self.goal_flags[0]:
            #pick_up_offset = 0.3
            pick_up_offset = 0.5
            action, cmds, self.state = self.pick_up_on_path(obs, self_pos, obj_pos, ob_key, self.state, pick_up_offset)
            if action == 'find_path':
                obs['comm'].append('path_found_'+str(self.avatar_id))
            if action == 'reset_arm':
                print('reset arm in flag 0')
                self.goal_flags[0] = True
                self.state = [False for i in range(8)]
                obs['comm'].remove('path_found_'+str(self.avatar_id))
                obs['comm'].append('agent_'+str(self.avatar_id)+' pickup_'+str(self.target_obj)+' frame_'+str(self.counter))
        elif self.bring2observer and not self.goal_flags[1]:
            # bring object to observer
            drop_offset = 1.7
            action, action_func, kwargs = self.drop_off(self_pos, fp1_pos, ob_key, drop_offset)
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
            if action_func == self.avatar.reset_arm:
                self.goal_flags[1] = True
                obs['comm'].append('agent_'+str(self.avatar_id)+' goal_'+str(self.target_obj)+' frame_'+str(self.counter))
        elif not self.goal_flags[1]:
            drop_offset = 0.2
            table_x_placement = 0.0
            table_positions = [[table_x_placement, 0.0, -1.14], [table_x_placement, 0.0, -2.85]]
            turn_targets = [[table_x_placement, 0.0, -6.0], [table_x_placement, 0.0, 6.0]]
            arm_id = self.avatar.static.joint_ids_by_name['magnet_left']
            arm_position = self.avatar.dynamic.joints[arm_id].position
            reach_targets = [[arm_position[0], arm_position[1]+0.4, arm_position[2]-0.35], [arm_position[0], arm_position[1]+0.4, arm_position[2]+0.33]]
            # go next to table in closest spot
            dist2table_pos = [distance.euclidean(self_pos, temp_pos) for temp_pos in table_positions]
            closest_spot = np.argmin(dist2table_pos)
            table_pos = np.array(table_positions[closest_spot])
            turn_target = np.array(turn_targets[closest_spot])
            reach_target = np.array(reach_targets[closest_spot])
            action, cmds, self.state, self.path_state = self.drop_on_obj(obs, self_pos, table_pos, turn_target, reach_target, ob_key, drop_offset, self.state, self.path_state)
            if action == 'find_path':
                obs['comm'].append('path_found_'+str(self.avatar_id))
            if action == 'reset_arm':
                print(f"Goal completed")
                self.state = []
                self.path_state = []
                self.goal_flags[1] = True
                obs['comm'].append('agent_'+str(self.avatar_id)+' goal_'+str(self.target_obj)+' frame_'+str(self.counter))
                # reset information about path
                obs['comm'].remove('path_found_'+str(self.avatar_id))
        elif all(self.goal_flags) and not self.reset_pos:
            # when done return to the original position
            arrived_offset = 0.2
            action, cmds, self.state = self.move_on_path(obs, self_pos, self.init_self_pos, self.state, arrived_offset)
            if action == 'find_path':
                obs['comm'].append('path_found_'+str(self.avatar_id))
            elif action == 'finished' and not self.reset_pos:
                obs['comm'].append('agent_'+str(self.avatar_id)+' resetpos frame_'+str(self.counter))
                self.reset_pos = True
                obs['comm'].remove('path_found_'+str(self.avatar_id))
        else:
            action = 'noop'
            self.prev_action = action
            cmds = []
        
        self.counter += 1
        return action, cmds
    
    
class MultistepMagnebotAgent(PremotorAgent):
    """A dummy magnebot example for using Magnebot agent
    as a subclass of PremotorAgent
    """
    def __init__(self, avatar_id, avatar, is_async, 
                 action_space, goal_sequence, bring2observer, collaborative=None, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "DummyMovingMagnebotAgent"
        self.counter = 0
        self.create_action_to_cmds_dict()
        self.bring2observer = bring2observer
        self.collaborative = collaborative # if collaborative gathering, leader or follower
        self.arm_reset = False
        self.carrying_obj = False
        self.start = False
        self.goal_sequence = goal_sequence
        self.goal_flags = [False for i in range(len(self.goal_sequence))]
        self.state = []
        self.path_state = []
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):   
        if self.collaborative and not self.start:
            if self.collaborative == 'leader':
                self.start = True
                obs['comm'] = ['start']
            elif self.collaborative == 'follower':
                if 'comm' in obs.keys():
                    self.start = 'first_pick_up' in obs['comm']
                else:
                    self.start = False
            self_pos = self.avatar.dynamic.transform.position
            self_pos = self_pos.astype(float)
            self.init_self_pos = self_pos
        elif not self.start:
            self.start = True
            obs['comm'] = ['start']
            self_pos = self.avatar.dynamic.transform.position
            self_pos = self_pos.astype(float)
            self.init_self_pos = self_pos
            
        # check for collisions
        if self.avatar.action.status == ActionStatus.collision or (self.prev_action == 'move_backward_collision' and self.avatar.action.status == ActionStatus.ongoing):
            collision_event = 'agent_'+str(self.avatar_id)+' collision frame_'+str(self.counter)
            print(collision_event)
            obs['comm'].append(collision_event)
            if self.prev_action != 'move_backward_collision':
                action = 'move_backward_collision'
                kwargs = {"distance": -1.5}
                action_func = self.avatar.move_by
                self.action_to_cmds_dict[action] = (action_func, kwargs)
                cmds = self.get_action_cmds(action)
            else:
                cmds = []
                action = 'noop'
            print('In collision loop', action)
                
            self.counter += 1
            return action, cmds
        
        #if obs['avsb'] and not all(self.goal_flags) and self.start and self.prev_action != 'move_backward_collision':
        if obs['avsb'] and not all(self.goal_flags) and self.start:
            obs_keys = list(obs['tran'].keys())
            current_goal = np.min([i for i, x in enumerate(self.goal_flags) if not x])
            target_obj = self.goal_sequence[current_goal]
            ob_key = obs_keys[target_obj]
            obj_pos = obs['tran'][ob_key]
            obj_pos = obj_pos.astype(float)
            self_pos = self.avatar.dynamic.transform.position
            self_pos = self_pos.astype(float)
            fp1_pos = np.array(obs['avsb']['fp1']['transform']['position'])
            if not self.bring2observer:
                table_key = obs_keys[-1]
                table_pos = obs['tran'][table_key]
                
            if not self.carrying_obj:
                pick_up_offset = 0.4
                action, cmds, self.state = self.pick_up_on_path(obs, self_pos, obj_pos, ob_key, self.state, pick_up_offset)
                if action == 'find_path':
                    obs['comm'].append('path_found_'+str(self.avatar_id))
                if action == 'reset_arm':
                    self.carrying_obj = True
                    if current_goal == 0 and self.collaborative == 'leader':
                        # if collaborative gathering, have leader signal when first obj is picked up
                        obs['comm'].append('first_pick_up')
                    obs['comm'].append('agent_'+str(self.avatar_id)+' pickup_'+str(target_obj)+' frame_'+str(self.counter))
                    self.state = [False for i in range(8)]
                    # reset information about path
                    obs['comm'].remove('path_found_'+str(self.avatar_id))
            elif self.bring2observer:
                drop_offset = 1.5
                # in collaborative mode, have agents deliver objects a different locations to prevent collisions
                if self.collaborative == 'follower':
                    fp1_pos[0] = fp1_pos[0] - 0.5
                elif self.collaborative == 'leader':
                    fp1_pos[0] = fp1_pos[0] + 0.5
                
                action, action_func, kwargs = self.drop_off(self_pos, fp1_pos, ob_key, drop_offset)
                self.action_to_cmds_dict[action] = (action_func, kwargs)
                cmds = self.get_action_cmds(action)
                if action_func == self.avatar.reset_arm:
                    print(f"Goal {current_goal} completed")
                    self.goal_flags[current_goal] = True
                    self.carrying_obj = False
                    obs['comm'].append('agent_'+str(self.avatar_id)+' goal_'+str(target_obj)+' frame_'+str(self.counter))
                    #obs['comm'].append('Goal_'+str(target_obj))
            elif self.collaborative == 'follower':
            # temp
            #elif not self.collaborative == 'follower':
                drop_offset = 0.12
                #table_x_placement = -1.0 + 1.0*current_goal
                #table_x_placement = -0.4 + 0.6*current_goal
                table_x_placement = -0.3 + 0.3*current_goal
                table_pos = np.array([table_x_placement, 0.0, -2.85])
                turn_target = np.array([table_x_placement, 0.0, 6.0])
                arm_id = self.avatar.static.joint_ids_by_name['magnet_left']
                arm_position = self.avatar.dynamic.joints[arm_id].position
                reach_target = arm_position
                reach_target[1] = reach_target[1] + 0.4
                reach_target[2] = reach_target[2] + 0.33
                action, cmds, self.state, self.path_state = self.drop_on_obj(obs, self_pos, table_pos, turn_target, reach_target, ob_key, drop_offset, self.state, self.path_state)
                #action, action_func, kwargs, self.state = self.drop_on_obj(self_pos, table_pos, turn_target, reach_target, ob_key, drop_offset, self.state)
                # self.action_to_cmds_dict[action] = (action_func, kwargs)
                # cmds = self.get_action_cmds(action)
                #if action_func == self.avatar.reset_arm:
                if action == 'find_path':
                    obs['comm'].append('path_found_'+str(self.avatar_id))
                if action == 'reset_arm':
                    print(f"Goal {current_goal} completed")
                    self.goal_flags[current_goal] = True
                    self.carrying_obj = False
                    self.state = []
                    self.path_state = []
                    obs['comm'].append('agent_'+str(self.avatar_id)+' goal_'+str(target_obj)+' frame_'+str(self.counter))
                    #obs['comm'].append('Goal_'+str(target_obj))
                    # reset information about path
                    obs['comm'].remove('path_found_'+str(self.avatar_id))
            else:
                drop_offset = 0.12
                #table_x_placement = -1.0 + 1.0*current_goal
                if self.collaborative == 'leader':
                    #table_x_placement = -0.0 + 0.5*current_goal 
                    table_x_placement = -0.2 + 0.5*current_goal 
                else:
                    table_x_placement = -0.8 + 0.6*current_goal 
                table_pos = np.array([table_x_placement, 0.0, -1.14])
                turn_target = np.array([table_x_placement, 0.0, -6.0])
                arm_id = self.avatar.static.joint_ids_by_name['magnet_left']
                arm_position = self.avatar.dynamic.joints[arm_id].position
                reach_target = arm_position
                reach_target[1] = reach_target[1] + 0.4
                reach_target[2] = reach_target[2] - 0.35
                # reach_target[1] = reach_target[1] + 0.5
                # reach_target[2] = reach_target[2] - 0.38
                action, cmds, self.state, self.path_state = self.drop_on_obj(obs, self_pos, table_pos, turn_target, reach_target, ob_key, drop_offset, self.state, self.path_state)
                #action, action_func, kwargs, self.state = self.drop_on_obj(self_pos, table_pos, turn_target, reach_target, ob_key, drop_offset, self.state)
                # self.action_to_cmds_dict[action] = (action_func, kwargs)
                # cmds = self.get_action_cmds(action)
                #if action_func == self.avatar.reset_arm:
                if action == 'find_path':
                    obs['comm'].append('path_found_'+str(self.avatar_id))
                if action == 'reset_arm':
                    print(f"Goal {current_goal} completed")
                    self.goal_flags[current_goal] = True
                    self.carrying_obj = False
                    self.state = []
                    self.path_state = []
                    obs['comm'].append('agent_'+str(self.avatar_id)+'goal_'+str(target_obj)+' frame_'+str(self.counter))
                    #obs['comm'].append('Goal_'+str(target_obj))
                    # reset information about path
                    obs['comm'].remove('path_found_'+str(self.avatar_id))
        elif all(self.goal_flags):
            self_pos = self.avatar.dynamic.transform.position
            # when done return to the original position
            arrived_offset = 0.2
            dist2target = distance.euclidean(self.init_self_pos, self_pos)
            if dist2target > arrived_offset:
                action, cmds, self.state = self.move_on_path(obs, self_pos, self.init_self_pos, self.state, arrived_offset)
                if action == 'find_path':
                    obs['comm'].append('path_found_'+str(self.avatar_id))
                if action == 'reset_arm':
                    self.goal_flags[current_goal] = True
                    self.carrying_obj = False
                    self.state = []
                    self.path_state = []
                    # reset information about path
                    obs['comm'].remove('path_found_'+str(self.avatar_id))
            else:
                cmds = []
                action = 'noop'
        elif obs['avsb'] and self.collaborative == 'follower':
            # when waiting to see what other agent is up to, rotate to see them
            non_leader_ids = [self.avatar_id, 'fp1']
            leader_id = [id for id in obs['avsb'].keys() if id not in non_leader_ids][0]
            target_pos = np.array(obs['avsb'][leader_id]['dynamic']['transform']['position'])
            action = 'rotate_to_target'
            kwargs = {"target": target_pos}
            self.action_to_cmds_dict[action] = (self.avatar.turn_to, kwargs)
            cmds = self.get_action_cmds(action)
        else:
            cmds = []
            action = 'noop'
            status = None
        
        self.counter += 1
        return action, cmds
    
    
# class MultistepMagnebotAgent2(PremotorAgent):
#     """works when you do not need to avoid obstacles
#     """
#     def __init__(self, avatar_id, avatar, is_async, 
#                  action_space, goal_sequence, bring2observer, collaborative=None, **kwargs):
#         super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
#         self.name = "DummyMovingMagnebotAgent"
#         self.counter = 0
#         self.create_action_to_cmds_dict()
#         self.bring2observer = bring2observer
#         self.collaborative = collaborative # if collaborative gathering, leader or follower
#         self.arm_reset = False
#         self.carrying_obj = False
#         self.goal_sequence = goal_sequence
#         self.goal_flags = [False for i in range(len(self.goal_sequence))]
#         self.state = []
        
#     def create_action_to_cmds_dict(self):
#         self.action_to_cmds_dict = {}
        
#     def act(self, obs):   
#         if self.collaborative:
#             if self.collaborative == 'leader':
#                 self.start = True
#             elif self.collaborative == 'follower':
#                 if 'comm' in obs.keys():
#                     self.start = 'first_pick_up' in obs['comm']
#                 else:
#                     self.start = False
#         else:
#             self.start = True
        
#         if obs['avsb'] and not all(self.goal_flags) and self.start:
#             obs_keys = list(obs['tran'].keys())
#             current_goal = np.min([i for i, x in enumerate(self.goal_flags) if not x])
#             target_obj = self.goal_sequence[current_goal]
#             ob_key = obs_keys[target_obj]
#             obj_pos = obs['tran'][ob_key]
#             obj_pos = obj_pos.astype(float)
#             self_pos = self.avatar.dynamic.transform.position
#             self_pos = self_pos.astype(float)
#             fp1_pos = np.array(obs['avsb']['fp1']['transform']['position'])
#             if not self.bring2observer:
#                 table_key = obs_keys[-1]
#                 table_pos = obs['tran'][table_key]

#             if not self.carrying_obj:
#                 # pick_up_offset = 0.2
#                 # action, action_func, kwargs = self.pick_up(self_pos, obj_pos, ob_key, pick_up_offset)
#                 # self.action_to_cmds_dict[action] = (action_func, kwargs)
#                 # cmds = self.get_action_cmds(action)
#                 # #cmds, status = self.get_action_cmds(action)
#                 #if action_func == self.avatar.reset_arm:
#                 pick_up_offset = 0.4
#                 action, cmds, self.state = self.pick_up_on_path(obs, self_pos, obj_pos, ob_key, self.state, pick_up_offset)
#                 if action == 'reset_arm':
#                     self.carrying_obj = True
#                     if current_goal == 0 and self.collaborative == 'leader':
#                         # if collaborative gathering, have leader signal when first obj is picked up
#                         obs['comm'] = ['first_pick_up']
#                     self.state = [False, False, False, False, False, False, False]
#             elif self.bring2observer:
#                 drop_offset = 1.5
#                 # in collaborative mode, have agents deliver objects a different locations to prevent collisions
#                 if self.collaborative == 'follower':
#                     fp1_pos[0] = fp1_pos[0] - 0.5
#                 elif self.collaborative == 'leader':
#                     fp1_pos[0] = fp1_pos[0] + 0.5
                
#                 action, action_func, kwargs = self.drop_off(self_pos, fp1_pos, ob_key, drop_offset)
#                 self.action_to_cmds_dict[action] = (action_func, kwargs)
#                 cmds = self.get_action_cmds(action)
#                 if action_func == self.avatar.reset_arm:
#                     print(f"Goal {current_goal} completed")
#                     self.goal_flags[current_goal] = True
#                     self.carrying_obj = False
#             elif self.collaborative == 'follower':
#                 drop_offset = 0.2
#                 #table_x_placement = -1.0 + 1.0*current_goal
#                 table_x_placement = -0.6 + 0.6*current_goal
#                 table_pos = np.array([table_x_placement, 0.0, -2.8])
#                 turn_target = np.array([table_x_placement, 0.0, 6.0])
#                 arm_id = self.avatar.static.joint_ids_by_name['magnet_left']
#                 arm_position = self.avatar.dynamic.joints[arm_id].position
#                 reach_target = arm_position
#                 reach_target[1] = reach_target[1] + 0.4
#                 reach_target[2] = reach_target[2] + 0.3
#                 action, action_func, kwargs, self.state = self.drop_on_obj(self_pos, table_pos, turn_target, reach_target, ob_key, drop_offset, self.state)
#                 self.action_to_cmds_dict[action] = (action_func, kwargs)
#                 cmds = self.get_action_cmds(action)
#                 if action_func == self.avatar.reset_arm:
#                     print(f"Goal {current_goal} completed")
#                     self.goal_flags[current_goal] = True
#                     self.carrying_obj = False
#                     self.state = []
#             else:
#                 drop_offset = 0.2
#                 #table_x_placement = -1.0 + 1.0*current_goal
#                 table_x_placement = -0.6 + 0.6*current_goal
#                 table_pos = np.array([table_x_placement, 0.0, -1.1])
#                 turn_target = np.array([table_x_placement, 0.0, -6.0])
#                 arm_id = self.avatar.static.joint_ids_by_name['magnet_left']
#                 arm_position = self.avatar.dynamic.joints[arm_id].position
#                 reach_target = arm_position
#                 reach_target[1] = reach_target[1] + 0.4
#                 reach_target[2] = reach_target[2] - 0.3
#                 action, action_func, kwargs, self.state = self.drop_on_obj2(self_pos, table_pos, turn_target, reach_target, ob_key, drop_offset, self.state)
#                 self.action_to_cmds_dict[action] = (action_func, kwargs)
#                 cmds = self.get_action_cmds(action)
#                 if action_func == self.avatar.reset_arm:
#                     print(f"Goal {current_goal} completed")
#                     self.goal_flags[current_goal] = True
#                     self.carrying_obj = False
#         elif all(self.goal_flags):
#             self_pos = self.avatar.dynamic.transform.position
#             # when done return to the center
#             if self.collaborative == 'follower':
#                 center_pos = np.array([1.0, 0.0, 0.0])
#             else:
#                 center_pos = np.array([0.0, 0.0, 1.0])
#             dist2target = distance.euclidean(center_pos, self_pos)
#             if dist2target > 0.5:
#                 action = "move"
#                 target = {"x": center_pos[0], "y": center_pos[1], "z": center_pos[2]}
#                 kwargs = {"target": target}
#                 self.action_to_cmds_dict[action] = (self.avatar.move_to, kwargs)
#                 cmds = self.get_action_cmds(action)
#             else:
#                 cmds = []
#                 action = 'noop'
#         elif obs['avsb'] and self.collaborative == 'follower':
#             # when waiting to see what other agent is up to, rotate to see them
#             non_leader_ids = [self.avatar_id, 'fp1']
#             leader_id = [id for id in obs['avsb'].keys() if id not in non_leader_ids][0]
#             target_pos = np.array(obs['avsb'][leader_id]['dynamic']['transform']['position'])
#             action = 'rotate_to_target'
#             kwargs = {"target": target_pos}
#             self.action_to_cmds_dict[action] = (self.avatar.turn_to, kwargs)
#             cmds = self.get_action_cmds(action)
#         else:
#             cmds = []
#             action = 'noop'
#             status = None
        
#         self.counter += 1
#         return action, cmds


class AdversarialGatheringMagnebotAgent(PremotorAgent):
    """An agent that removes objects on the table placed by gathering agent
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, obs2move, following_id, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "AdversarialGatheringMagnebotAgent"
        self.counter = 0
        self.create_action_to_cmds_dict()
        self.obs2move = obs2move
        self.following_id = following_id
        self.goal_incomplete = False
        # self.goal_flags = [False, False]
        self.state = [False for i in range(10)]
        self.path_state = []
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):  
        obs_keys = list(obs['tran'].keys())
        obj_pos_list = [obs['tran'][ob_key] for ob_key in obs_keys]
        #obj_pos_list = [obs['tran'][obs_keys[obj_num]] for obj_num in self.obs2move]
        table_key = obs_keys[-1]
        table_pos = obs['tran'][table_key]
        self_pos = self.avatar.dynamic.transform.position
        obj_on_table = []
        
        if self.counter == 0:
            self.obj_init_pos = {}
            for obj_num in self.obs2move:
                ob_key = obs_keys[obj_num]
                self.obj_init_pos[ob_key] = obj_pos_list[obj_num]
                
        # for obj_num in self.obs2move:
        #     temp_obj_pos = obj_pos_list[obj_num]
        #     dist2table = distance.euclidean(temp_obj_pos[2], table_pos[2])
        #     if obj_num == 0:
        #         print('Dist',dist2table)
        #         print(temp_obj_pos)
        #     if temp_obj_pos[1] > 0.33 and temp_obj_pos[1] < 0.52 and dist2table < 0.1:
        #         print('Here',obj_num)
        #         obj_on_table.append(obj_num)
                
        if 'comm' in obs:
            for entry in obs['comm']:
                if entry[:4] == 'Goal':
                    temp_obj = int(entry[-1])
                    obj_on_table.append(temp_obj)
                
        if len(obj_on_table) > 0 or self.goal_incomplete:
            ob_key = obs_keys[obj_on_table[0]]
            obj_pos = obj_pos_list[obj_on_table[0]]
            table_x_placement = obj_pos[0]
            #target_pos = np.array([table_x_placement, 0.0, -2.8])
            target_pos = np.array([table_x_placement, 0.0, -2.75])
            turn_target = np.array([table_x_placement, 0.0, 6.0])
                
        if len(obj_on_table) == 0 and not self.goal_incomplete:
            #if self.counter == 0:
            #temp stop rotating to look at other agent, because rotation causes problems
            if self.counter > -1:
                action = 'noop'
                cmds = []
            else:
                following_pos = np.array(obs['avsb'][self.following_id]['dynamic']['transform']['position'])
                action = 'rotate_to_target'
                kwargs = {"target": following_pos}
                self.action_to_cmds_dict[action] = (self.avatar.turn_to, kwargs)
                cmds = self.get_action_cmds(action)
        elif not self.state[0]:
            self.goal_incomplete = True
            arrived_offset = 0.3
            action, cmds, self.path_state = self.move_on_path(obs, self_pos, target_pos, self.path_state, arrived_offset)
            if action == 'find_path':
                obs['comm'].append('path_found_'+str(self.avatar_id))
            if all(self.path_state) and len(self.path_state) > 0:
                self.state[0] = True
                self.path_state = []
                obs['comm'].remove('path_found_'+str(self.avatar_id))
        elif not self.state[1]:
            # turn to face the table at a 90 degree angle
            action = 'turn_to'
            kwargs = {"target": turn_target, "aligned_at": 0.5}
            action_func = self.avatar.turn_to
            if self.prev_action == action:
                action_status = self.avatar.action.status
                if action_status == ActionStatus.success:
                    self.state[1] = True
                elif action_status == ActionStatus.collision:
                    #rotating may cause a collision
                    action = 'move_backward'
                    kwargs = {"distance": -0.2}
                    action_func = self.avatar.move_by
                else:
                    self.state[1] = False
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
        elif not self.state[2]:
            # in case you moved backwards while rotating, move forward to target again
            dist2target = np.abs(target_pos[2] - self_pos[2])
            #if dist2target > drop_offset:
            if dist2target > 0.05:
                action = "move_forward"
                kwargs = {"distance": dist2target, "arrived_at": 0.01}
                action_func = self.avatar.move_by
                if self.prev_action == action:
                    self.state[2] = self.avatar.action.status == ActionStatus.success
                # moving forward action may cause collision if holding object touches table
                # if so, move ahead to next action
                if self.avatar.action.status == ActionStatus.collision:
                    self.state[2] = True
                self.action_to_cmds_dict[action] = (action_func, kwargs)
                cmds = self.get_action_cmds(action)
            else:
                self.state[2] = True
                action = "raise_torso"
                y_pos = 3.0
                kwargs = {"height": y_pos}
                action_func = self.avatar.slide_torso
                if self.prev_action == action:
                    self.state[3] = self.avatar.action.status == ActionStatus.success
                self.action_to_cmds_dict[action] = (action_func, kwargs)
                cmds = self.get_action_cmds(action)
        elif not self.state[3]:
            action = "raise_torso"
            y_pos = 3.0
            kwargs = {"height": y_pos}
            action_func = self.avatar.slide_torso
            if self.prev_action == action:
                self.state[3] = self.avatar.action.status == ActionStatus.success
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
        elif not self.state[4]:
            action = 'reach'
            reach_target = obj_pos
            kwargs = {"target": reach_target, "arm": Arm.left, "absolute": True, "arrived_at": 0.125}
            action_func = self.avatar.reach_for
            if self.prev_action == action:
                self.state[4] = self.avatar.action.status == ActionStatus.success
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
        elif not self.state[5]:
            obj_held = bool(self.avatar.dynamic.held[Arm.left])
            if not obj_held:
                action = "grab"
                kwargs = {"target": ob_key, "arm": Arm.left}
                action_func = self.avatar.grasp
            else:
                arm_id = self.avatar.static.joint_ids_by_name['magnet_left']
                arm_position = self.avatar.dynamic.joints[arm_id].position
                reach_target = arm_position
                #reach_target[1] = reach_target[1] + 0.8
                reach_target[1] = reach_target[1] + 0.2
                action = 'reach'
                kwargs = {"target": reach_target, "arm": Arm.left, "absolute": True, "arrived_at": 0.125}
                action_func = self.avatar.reach_for
                if self.prev_action == action:
                    self.state[5] = self.avatar.action.status == ActionStatus.success
                if self.avatar.action.status == ActionStatus.failed_to_reach or action == "reset_arm":
                    action = "reset_arm"
                    kwargs = {"arm": Arm.left}
                    action_func = self.avatar.reset_arm
                action = "raise_torso"
                #y_pos = 4.5
                y_pos = 2.0
                kwargs = {"height": y_pos}
                action_func = self.avatar.slide_torso
                if self.prev_action == action:
                    self.state[5] = self.avatar.action.status == ActionStatus.success
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
            #     action = "reset_arm"
            #     kwargs = {"arm": Arm.left}
            #     action_func = self.avatar.reset_arm
            # self.action_to_cmds_dict[action] = (action_func, kwargs)
            # cmds = self.get_action_cmds(action)
            # if action == "reset_arm" and self.prev_action == action:
            #     action_status = self.avatar.action.status
            #     if action_status == ActionStatus.success:
            #         self.state[5] = True
        elif not self.state[6]:
            action = "move_back"
            kwargs = {"distance": -0.5}
            action_func = self.avatar.move_by
            if self.prev_action == action:
                self.state[6] = self.avatar.action.status == ActionStatus.success
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
        elif not self.state[7]:
            action = "reset_arm"
            kwargs = {"arm": Arm.left}
            action_func = self.avatar.reset_arm
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
            if action == "reset_arm" and self.prev_action == action:
                self.state[7] = True
                # self.state[7] = self.avatar.action.status == ActionStatus.success
                # print(self.avatar.action.status)
        elif not self.state[8]:
            og_pos = self.obj_init_pos[ob_key]
            arrived_offset = 0.3
            action, cmds, self.path_state = self.move_on_path(obs, self_pos, og_pos, self.path_state, arrived_offset)
            if action == 'find_path':
                obs['comm'].append('path_found_'+str(self.avatar_id))
            if all(self.path_state) and len(self.path_state) > 0:
                self.state[8] = True
                self.path_state = []
                obs['comm'].remove('path_found_'+str(self.avatar_id))
        elif not self.state[9]:
            obj_held = bool(self.avatar.dynamic.held[Arm.left])
            if obj_held:
                action = "drop"
                kwargs = {"target": ob_key, "arm": Arm.left}
                action_func = self.avatar.drop
                self.action_to_cmds_dict[action] = (action_func, kwargs)
                cmds = self.get_action_cmds(action)
            else:
                self.state[9] = True
                action = 'noop'
                cmds = []
            # else:
            #     action = "reset_arm"
            #     kwargs = {"arm": Arm.left}
            #     action_func = self.avatar.reset_arm
            if self.prev_action == action:
                self.state[9] = self.avatar.action.status == ActionStatus.success
            if self.state[9]:
                self.goal_incomplete = False
                self.state = [False for i in range(10)]
                target_obj = obj_on_table[0]
                obs['comm'].remove('Goal_'+str(target_obj))
                
        self.counter += 1
        return action, cmds
    
class AdversarialGatheringMagnebotAgentNoTable(PremotorAgent):
    """An agent that removes objects on the table placed by gathering agent
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, obs2move, following_id, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "AdversarialGatheringMagnebotAgentNoTable"
        self.counter = 0
        self.create_action_to_cmds_dict()
        self.obs2move = obs2move
        self.following_id = following_id
        self.goal_incomplete = False
        # self.goal_flags = [False, False]
        self.state = [False for i in range(4)]
        self.path_state = []
        self.other_agent_init_pos = None
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):  
        obs_keys = list(obs['tran'].keys())
        obj_pos_list = [obs['tran'][ob_key] for ob_key in obs_keys]
        self_pos = self.avatar.dynamic.transform.position
        obj_at_goal = []
        if obs['avsb']:
            fp1_pos = np.array(obs['avsb']['fp1']['transform']['position'])
            if self.other_agent_init_pos is None:
                avsb_keys = list(obs['avsb'].keys())
                self.other_agent_id = [key for key in avsb_keys if key not in ['fp1', self.avatar_id]][0]
                self.other_agent_init_pos = TDWUtils.vector3_to_array(obs['avsb'][self.other_agent_id]['initial_position'])
            # has gathering agent returned to initial position to avoid collisions?
            other_agent_pos = obs['avsb'][self.other_agent_id]['dynamic']['transform']['position']
            other_agent_back = distance.euclidean(other_agent_pos, self.other_agent_init_pos) < 1.0
            
        if self.counter == 0:
            self.init_self_pos = self_pos
            self.obj_init_pos = {}
            for obj_num in self.obs2move:
                print(obs_keys)
                ob_key = obs_keys[obj_num]
                self.obj_init_pos[ob_key] = obj_pos_list[obj_num]
                
        if 'comm' in obs:
            for entry in obs['comm']:
                if 'goal' in entry:
                    match = re.search('goal_(\d+)', entry)
                    temp_obj = int(match.group(1))
                    if temp_obj not in obj_at_goal:
                        obj_at_goal.append(temp_obj)
                    
                if 'resetpos' in entry and len(obj_at_goal) > 0  and not self.goal_incomplete and other_agent_back:
                    obs['comm'].append('agent_'+str(self.avatar_id)+' start-adversary_'+str(obj_at_goal[0])+' frame_'+str(self.counter))
                    self.goal_incomplete = True
        
        if len(obj_at_goal) > 0 or self.goal_incomplete:
            ob_key = obs_keys[obj_at_goal[0]]
            target_pos = obj_pos_list[obj_at_goal[0]]
            # check if adversary already returned obj to original position, otherwise gather it from goal location
            if distance.euclidean(self.obj_init_pos[ob_key], target_pos) < 1.0 and all(self.state):
                self.goal_incomplete = False
            # check if object is at goal location
            if distance.euclidean(target_pos, fp1_pos) < 1.5 and all(self.state) and other_agent_back:
                self.goal_incomplete = True
                self.state = [False for i in range(4)]
                self.path_state = []

        if len(obj_at_goal) == 0 or not self.goal_incomplete:
            #if self.counter == 0:
            #temp stop rotating to look at other agent, because rotation causes problems
            if self.counter > -1:
                action = 'noop'
                cmds = []
            else:
                following_pos = np.array(obs['avsb'][self.following_id]['dynamic']['transform']['position'])
                action = 'rotate_to_target'
                kwargs = {"target": following_pos}
                self.action_to_cmds_dict[action] = (self.avatar.turn_to, kwargs)
                cmds = self.get_action_cmds(action)
        elif not self.state[0]:
            #self.goal_incomplete = True
            pick_up_offset = 0.2
            action, cmds, self.path_state = self.pick_up_on_path(obs, self_pos, target_pos, ob_key, self.path_state, pick_up_offset)
            if action == 'find_path':
                obs['comm'].append('path_found_'+str(self.avatar_id))
            if action == 'reset_arm':
                self.state[0] = True
                self.path_state = []
                obs['comm'].remove('path_found_'+str(self.avatar_id))
                obs['comm'].append('agent_'+str(self.avatar_id)+' pickup_'+str(obj_at_goal[0])+' frame_'+str(self.counter))
        elif not self.state[1]:
            # bring object back to original location
            og_pos = self.obj_init_pos[ob_key]
            arrived_offset = 0.3
            action, cmds, self.path_state = self.move_on_path(obs, self_pos, og_pos, self.path_state, arrived_offset)
            if action == 'find_path':
                obs['comm'].append('path_found_'+str(self.avatar_id))
            if all(self.path_state) and len(self.path_state) > 0:
                self.state[1] = True
                self.path_state = []
                obs['comm'].remove('path_found_'+str(self.avatar_id))
        elif not self.state[2]:
            obj_held = bool(self.avatar.dynamic.held[Arm.left])
            if obj_held:
                action = "drop"
                kwargs = {"target": ob_key, "arm": Arm.left}
                action_func = self.avatar.drop
                self.action_to_cmds_dict[action] = (action_func, kwargs)
                cmds = self.get_action_cmds(action)
            # else:
            #     self.state[9] = True
            #     action = 'noop'
            #     cmds = []
            else:
                action = "reset_arm"
                kwargs = {"arm": Arm.left}
                action_func = self.avatar.reset_arm
                self.action_to_cmds_dict[action] = (action_func, kwargs)
                cmds = self.get_action_cmds(action)
                self.state[2] = True
                # if self.prev_action == action:
                #     self.state[2] = self.avatar.action.status == ActionStatus.success
        elif not self.state[3]:
            # when done return to the original position
            arrived_offset = 0.2
            action, cmds, self.path_state = self.move_on_path(obs, self_pos, self.init_self_pos, self.path_state, arrived_offset)
            if action == 'find_path':
                obs['comm'].append('path_found_'+str(self.avatar_id))
            elif action == 'finished':
                print('FINISHED RETURNING')
                self.state[3] = True
                self.goal_incomplete = False
                obs['comm'].append('agent_'+str(self.avatar_id)+' return_'+str(obj_at_goal[0])+' frame_'+str(self.counter))
                obs['comm'].remove('path_found_'+str(self.avatar_id))
                # if self.state[2]:
                #     self.goal_incomplete = False
                #     self.state = [False for i in range(3)]
                #     obs['comm'].append('agent_'+str(self.avatar_id)+' return_'+str(obj_at_goal[0])+' frame_'+str(self.counter))
        else:
            action = 'noop'
            cmds = []
        
        self.counter += 1
        return action, cmds
        
    
class MagnebotAsyncAgent(Agent):
    """A DummyAsyncAgent that can only rotate left and right.

    A key characteristic of async agents is that act() takes an action taken by a
    user then convert it to TDW commands.
    """
    def __init__(self, avatar_id, avatar, is_async, action_space):
        self.name = 'DummyAsyncAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)        
        self.create_action_to_cmds_dict()

    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}

    def act(self, action):
        if action == 'r-left':
            kwargs = {"angle": -30}
        elif action == 'r-right':
            kwargs = {"angle": 30}
            
        self.action_to_cmds_dict[action] = (self.avatar.turn_by, kwargs)
        cmds = self.get_action_cmds(action)            
        return action, self.action_to_cmds_dict[action]
    
    
class DummyAsyncAgent(Agent):
    """A DummyAsyncAgent that can only rotate left and right.

    A key characteristic of async agents is that act() takes an action taken by a
    user then convert it to TDW commands.
    """
    def __init__(self, avatar_id, avatar, is_async, action_space):
        self.name = 'DummyAsyncAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)        
        self.create_action_to_cmds_dict()

    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {
            'r-left': [{"$type": "rotate_avatar_by", "angle": -30, "avatar_id": self.avatar_id}],
            'r-right': [{"$type": "rotate_avatar_by", "angle": 30, "avatar_id": self.avatar_id}]}

    def act(self, action):
        return action, self.action_to_cmds_dict[action]

        
class DummyAgent(Agent):
    """A DummyAgent that just rotate left and right regardless of its observation
    """
    def __init__(self, avatar_id, avatar, is_async, action_space):
        self.name = 'DummyAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)        
        self.create_action_to_cmds_dict()
        self.counter = 0
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {
            'r-right': [{"$type": "rotate_avatar_by", "angle": -30, "avatar_id": self.avatar_id}],
            'r-left': [{"$type": "rotate_avatar_by", "angle": 30, "avatar_id": self.avatar_id}]
        }
        
    def act(self, obs):
        commands = []
        actions = ['r-right' for _ in range(10)] + ['r-left' for _ in range(10)]
        action = actions[self.counter % len(actions)]
        self.counter += 1
        return action, self.action_to_cmds_dict[action]

    
class GatheringAgent(Agent):
    def __init__(self, avatar_id, avatar, is_async, action_space, target_obj):
        self.name = 'ActorAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)        
        self.target_obj = target_obj
        self.step_size = 0.005

    def act(self, obs):
        obs_keys = list(obs['tran'].keys())
        ob_key = obs_keys[self.target_obj]
        obj_pos = obs['tran'][ob_key]
        
        commands = []
        action = 'move'
        commands.append(
            {"$type": "move_avatar_towards_position",
            "position": {"x": float(obj_pos[0]), "y": float(obj_pos[1]), "z": float(obj_pos[2])},
            "magnitude": self.step_size,
            "avatar_id": self.avatar_id})
        return action, commands
    
    
class MultistepAgent(Agent):
    def __init__(self, avatar_id, avatar, is_async, action_space, goal_sequence):
        self.name = 'ActorAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)        
        self.goal_sequence = goal_sequence
        self.goal_flags = [False for i in range(len(self.goal_sequence))]
        self.step_size = 0.005
        self.dist2objs = 0.5
        self.repeat_sequence = True
    
    def act(self, obs):
        if obs['avsb'] and not all(self.goal_flags):
            obs_keys = list(obs['tran'].keys())
            current_goal = np.min([i for i, x in enumerate(self.goal_flags) if not x])
            target_obj = self.goal_sequence[current_goal]
            ob_key = obs_keys[target_obj]
            obj_pos = obs['tran'][ob_key]
            self_pos = np.array(obs['avsb'][self.avatar_id]['transform']['position'])
            dist2target = distance.euclidean(obj_pos, self_pos)
            is_moving = obs['avsb'][self.avatar_id]['is_moving']
            
            commands = []

            if dist2target > self.dist2objs:
                action = 'move'
                commands.append(
                    {"$type": "move_avatar_towards_position",
                    "position": {"x": float(obj_pos[0]), "y": float(obj_pos[1]), "z": float(obj_pos[2])},
                    "magnitude": self.step_size,
                    "avatar_id": self.avatar_id})
            else:
                # goal is completed when you get close enough to object
                # set flag to true so next goal can be completed
                print(f"Goal {current_goal} completed")
                self.goal_flags[current_goal] = True
                action = 'noop'
        else:
            commands = []
            action = 'noop'
            
        if all(self.goal_flags) and self.repeat_sequence:
            self.goal_flags = [False for i in range(len(self.goal_sequence))]
            
        return action, commands
    
    
class ObserverAgent(Agent):
    def __init__(self, avatar_id, avatar, is_async, action_space):
        self.name = 'ObserverAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)        
        self.create_action_to_cmds_dict()
        self.counter = 0
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {
            'r-right': [{"$type": "rotate_avatar_by", "angle": -30, "avatar_id": self.avatar_id}],
            'r-left': [{"$type": "rotate_avatar_by", "angle": 30, "avatar_id": self.avatar_id}],
            'noop': [{"$type": "rotate_avatar_by", "angle": 0, "avatar_id": self.avatar_id}]
        }
        
    def act(self, obs):
        commands = []
        action = 'noop'
        self.counter += 1
        return action, self.action_to_cmds_dict[action]     
    
    
class GazeFetchObserverAgent(Agent):
    def __init__(self, avatar_id, avatar, is_async, action_space, target_obj):
        self.name = 'GazeFetchObserverAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)        
        self.counter = 0
        self.target_obj = target_obj
        self.goal_flags = [False, False, False]
        
    def act(self, obs):
        if self.counter == 0:
            self.counter += 1
            return 'noop', []
        
        obs_keys = list(obs['tran'].keys())
        ob_key = obs_keys[self.target_obj]
        obj_pos = obs['tran'][ob_key]
        self_pos = np.array(obs['avsb']['fp1']['transform']['position'])
        self_angle = obs['avsb']['fp1']['transform']['rotation']
        if 'comm' in obs.keys():
            # contingency met
            self.goal_flags[0] = 'joint_attention' in obs['comm']
            
        if self.goal_flags[0] and not self.goal_flags[1]:
            # calculate angle of where target object is to rotate there
            x_diff = obj_pos[0] - self_pos[0]
            z_diff = obj_pos[2] - self_pos[2]
            target_angle = math.degrees(np.arctan(x_diff / z_diff))
            cmds = [{'$type': 'rotate_avatar_by', 'angle': target_angle, 'avatar_id': self.avatar_id}]
            action = 'rotate_to_target'
            self.goal_flags[1] = True
            obs['comm'].append('gazed_at_obj')
        elif self.goal_flags[0] and self.goal_flags[1] and not self.goal_flags[2]:
            self.avatar.look_at(ob_key)
            action = 'look_at_obj'
            cmds = []
            # contingency met
            self.goal_flags[2] = 'obj_delivered' in obs['comm']
        elif all(self.goal_flags):
            self.avatar.look_at(3434)
            action = 'look_at_actor'
            cmds = []
        else:
            action = 'noop'
            cmds = []
            
        self.counter += 1
        return action, cmds
    
    
class GazeFetchMagnebotAgent(PremotorAgent):
    """A dummy magnebot example for using Magnebot agent
    as a subclass of PremotorAgent
    """
    def __init__(self, avatar_id, avatar, is_async, action_space, target_obj, bring2observer, **kwargs):
        super().__init__(avatar_id, avatar, is_async, action_space, **kwargs)
        self.name = "DummyMovingMagnebotAgent"
        self.counter = 0
        self.create_action_to_cmds_dict()
        self.target_obj = target_obj
        self.bring2observer = bring2observer
        self.arm_reset = False
        self.goal_flags = [False, False, False, False]
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):
        if self.counter == 0:
            self.counter += 1
            return 'noop', []
            
        obs_keys = list(obs['tran'].keys())
        ob_key = obs_keys[self.target_obj]
        obj_pos = obs['tran'][ob_key]
        self_pos = self.avatar.dynamic.transform.position
        
        fp1_pos = np.array(obs['avsb']['fp1']['transform']['position'])
            
        if not self.goal_flags[0]:
            # rotate to look at observer
            obs['comm'] = ['start']
            # rotate to observer
            # self.avatar.turn_to(fp1_pos)
            # cmds = []
            action = 'rotate_to_target'
            kwargs = {"target": fp1_pos}
            self.action_to_cmds_dict[action] = (self.avatar.turn_to, kwargs)
            cmds = self.get_action_cmds(action)
            
            if self.avatar.action.status == ActionStatus.success:
                self.goal_flags[0] = True
                obs['comm'].append('joint_attention')
        elif not self.goal_flags[1]:
            # wait until observer stares at the object it wants
            self.goal_flags[1] = 'gazed_at_obj' in obs['comm']
            cmds = []
            action = 'noop'
        elif not self.goal_flags[2]:
            # pick up object
            action, action_func, kwargs = self.pick_up(self_pos, obj_pos, ob_key)
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
            if action_func == self.avatar.reset_arm:
                self.goal_flags[2] = True
        elif self.bring2observer and not self.goal_flags[3]:
            # bring object to observer
            drop_offset = 1.5
            action, action_func, kwargs = self.drop_off(self_pos, fp1_pos, ob_key, drop_offset)
            self.action_to_cmds_dict[action] = (action_func, kwargs)
            cmds = self.get_action_cmds(action)
            if action_func == self.avatar.reset_arm:
                self.goal_flags[3] = True
                obs['comm'].append('obj_delivered')
        elif all(self.goal_flags):
            self_pos = self.avatar.dynamic.transform.position
            # when done return to the center
            center_pos = np.array([0, 0, 0])
            dist2target = distance.euclidean(center_pos, self_pos)
            if dist2target > 0.5:
                action = "move"
                target = {"x": 0, "y": 0, "z": 0}
                kwargs = {"target": target}
                self.action_to_cmds_dict[action] = (self.avatar.move_to, kwargs)
                cmds = self.get_action_cmds(action)
            else:
                cmds = []
                action = 'noop'
        else:
            cmds = []
            action = 'noop'
            
        self.counter += 1
        return action, cmds
    
    
class RandomAgent(Agent):
    def __init__(self, avatar_id, avatar, is_async, action_space, save_commands, observer_id, mime_id):
        self.name = 'RandomAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)        
        self.step_size = 0.1
        self.path_len = 2
        self.counter = 0
        self.save_commands = save_commands
        self.observer_id = observer_id
        self.mime_id = mime_id
        
    def act(self, obs):
        illegal_action = True
        if obs['avsb']:
            self.avatar_position = obs['avsb'][self.avatar_id]['transform']['position']
            self.observer_position = obs['avsb'][self.observer_id]['transform']['position']
            self.mime_position = obs['avsb'][self.mime_id]['transform']['position']
            self.occ_map = obs['occ_map']
            
            # repeat same transformation until avatar is close to destination
            if self.counter > 0:
                target_pos = np.array(self.new_avatar_center)[[0,2]]
                current_pos = np.array(self.avatar_position)[[0,2]]
            if self.counter == 0 or np.allclose(target_pos, current_pos, atol=1e-4):
                while illegal_action:
                    new_avatar_center = self.get_random_position()
                    # make sure next action is in bounds
                    x_ind, z_ind = self.find_location_occupancy(self.occ_map, new_avatar_center[0], new_avatar_center[2])
                    # find occupancy of this index and adjacent indices
                    inds2check = [(x_ind-1,z_ind),(x_ind,z_ind-1),(x_ind,z_ind),(x_ind+1,z_ind),(x_ind,z_ind+1)]
                    oob = False
                    for x,z in inds2check:
                        occupancy = self.occ_map.occupancy_map[x, z]
                        if occupancy != 0:
                            oob = True
                            break
                        
                    if oob:
                        illegal_action = True
                        continue
                    else:
                        # make sure you don't run into observer or mimicry agent
                        illegal_action = np.allclose(new_avatar_center, self.observer_position, atol=5e-1) or np.allclose(new_avatar_center, self.mime_position, atol=5e-1)
                self.new_avatar_center = new_avatar_center
                
                # if a separate mimicry agent is copying these random actions, save them
                if self.save_commands:
                    self.save_command()
                
            action = 'move_avatar_towards_position'
            cmd = {"$type": "move_avatar_towards_position",
                  "position": {"x": self.new_avatar_center[0], "y": self.new_avatar_center[1], "z": self.new_avatar_center[2]},
                  "speed": self.step_size,
                  "avatar_id": self.avatar_id}
            commands = [cmd]
            self.counter += 1
        else:
            action = 'noop'
            commands = []

        return action, commands
    
    def get_random_position(self):
        rand_x = self.avatar_position[0] + random.uniform(-self.path_len, self.path_len)
        rand_z = self.avatar_position[2] + random.uniform(-self.path_len, self.path_len)
        new_avatar_center = [rand_x, self.avatar_position[1], rand_z]
        
        return new_avatar_center
    
    def find_location_occupancy(self, occupancy_map, location_x, location_z):
        cell_size = 0.5
        occ_shape = occupancy_map.occupancy_map.shape
        xmin, zmin = occupancy_map.get_occupancy_position(0, 0)
        xmax, zmax = occupancy_map.get_occupancy_position(occ_shape[0]-1, occ_shape[1]-1)
        
        xbins = np.arange(xmin, xmax+cell_size, cell_size)
        zbins = np.arange(zmin, zmax+cell_size, cell_size)
        
        x_ind = self.find_nearest_ind(xbins, location_x)
        z_ind = self.find_nearest_ind(zbins, location_z)
        
        return x_ind, z_ind
    
    def find_nearest_ind(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def save_command(self):
        script_path = os.path.abspath(__file__)
        save_file = script_path[:-8]+'agent_log/mimicry_actions.csv'
        new_pos = np.array(self.new_avatar_center).reshape(1, -1)
        np.savetxt(save_file, new_pos, delimiter=",")
    
    
class MimicAgent(Agent):
    def __init__(self, avatar_id, avatar, is_async, action_space, following_id):
        self.name = 'MimicAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)        
        self.step_size = 0.1
        self.counter = 0
        self.delay = 10
        self.following_id = following_id
        self.leader_goals = []
        
    def act(self, obs):
        commands = []
        if obs['avsb'] and self.counter < self.delay:
            # how far away are we from agent we are mimicking
            self.avatar_position = obs['avsb'][self.avatar_id]['transform']['position']
            self.leader_pos = obs['avsb'][self.following_id]['transform']['position']
            self.offset = np.array(self.avatar_position) - np.array(self.leader_pos)
            action = 'noop'
            leader_goal = self.load_command()
            self.leader_goals.append(leader_goal)
            self.counter += 1
        elif obs['avsb']:
            self.avatar_position = obs['avsb'][self.avatar_id]['transform']['position']
            leader_goal = self.load_command()
            self.leader_goals.append(leader_goal)
            
            if self.counter == self.delay or np.allclose(self.new_avatar_center, self.avatar_position, atol=1e-4):
                delayed_goal = self.leader_goals[-self.delay]
                new_avatar_center = np.array(delayed_goal) + self.offset
                self.new_avatar_center = new_avatar_center.tolist()

            action = 'move_avatar_towards_position'
            cmd = {"$type": "move_avatar_towards_position",
              "position": {"x": self.new_avatar_center[0], "y": self.new_avatar_center[1], "z": self.new_avatar_center[2]},
              "speed": self.step_size,
              "avatar_id": self.avatar_id}
            commands.append(cmd)
            self.counter += 1
        else:
            action = 'noop'

        return action, commands
    
    def load_command(self):
        script_path = os.path.abspath(__file__)
        save_file = script_path[:-8]+'agent_log/mimicry_actions.csv'
        leader_goal = np.genfromtxt(save_file, delimiter=",")
        
        return leader_goal
    
    
class MagnebotMimicAgent(PremotorAgent):
    def __init__(self, avatar_id, avatar, is_async, action_space, following_id):
        self.name = 'MagnebotMimicAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)        
        self.step_size = 0.1
        self.counter = 0
        self.delay = 10
        self.following_id = following_id
        self.create_action_to_cmds_dict()
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):
        if self.counter == 0:
            obs['comm'] = ['start']
            action = 'noop'
            cmds = []
        elif self.counter < self.delay:
            # how far away are we from agent we are mimicking
            self.avatar_position = self.avatar.dynamic.transform.position
            self.leader_pos = obs['avsb'][self.following_id]['dynamic']['transform']['position']
            self.offset = np.array(self.avatar_position) - np.array(self.leader_pos)
            action = 'noop'
            cmds = []
        else:
            delayed_target = obs['comm'][-1]
            if isinstance(delayed_target, np.ndarray):
                target_pos = delayed_target + self.offset
                arrived_offset = 0.3
                kwargs = {"target": target_pos, "arrived_offset": arrived_offset}
                action = "move"
                self.action_to_cmds_dict[action] = (self.avatar.move_to, kwargs)        
                cmds = self.get_action_cmds(action)
            else:
                action = 'noop'
                cmds = []
        self.counter += 1
    
        return action, cmds
        
    
class ChasingAgent(Agent):
    def __init__(self, avatar_id, avatar, is_async, action_space, following_id):
        self.name = 'ChasingAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)       
        self.following_id = following_id
        self.step_size = 0.4
        self.counter = 0
        
    def act(self, obs):
        if obs['avsb']:
            target_pos = obs['avsb'][self.following_id]['transform']['position']
            
            commands = []
            action = 'move'
            commands.append(
                {"$type": "move_avatar_towards_position",
                "position": {"x": float(target_pos[0]), "y": float(target_pos[1]), "z": float(target_pos[2])},
                "speed": self.step_size,
                "avatar_id": self.avatar_id})
        else:
            action = 'noop'
            commands = []
            
        return action, commands
    

class RunnerAgent(Agent):
    def __init__(self, avatar_id, avatar, is_async, action_space, follower_id):
        self.name = 'RunnerAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)       
        self.follower_id = follower_id
        self.step_size = 0.7
        self.counter = 0
        self.target_rotation = 0.
        self.debug = False
        self.avatar_position = [0., 0., 0.]
        self.new_runner_pos = [0., 0., 0.]
        # to do make this parametrized
        self.x_limits = (-5, 5)
        self.z_limits = (-5, 5)
        
    def get_escape_angle(self, obs):
        if isinstance(self.follower_id, str):
            chaser_pos = obs['avsb'][self.follower_id]['transform']['position']
        elif isinstance(self.follower_id, int):
            chaser_pos = obs['avsb'][self.follower_id]['dynamic']['transform']['position']
        runner_pos = obs['avsb'][self.avatar_id]['transform']['position']
          
        x_d = chaser_pos[0] - runner_pos[0]
        z_d = chaser_pos[2] - runner_pos[2]
        
        # Getting the angle
        hyp_d = math.sqrt(x_d**2 + z_d**2)
        angle = math.degrees(np.arcsin(abs(z_d)/hyp_d)) # pythagoras; fyi: math.sin takes angle in radians
        
        # Direction that chaser is in
        if z_d >= 0.0 and x_d >= 0.0:
            #direction = 90. - angle
            corrected_angle = angle
        elif z_d >= 0.0 and x_d < 0.0:
            #direction = -(90. - angle)
            corrected_angle = 180 - angle
        elif z_d < 0.0 and x_d >= 0.0:
            #direction = 90. + angle
            corrected_angle = 360 - angle
        elif z_d < 0.0 and x_d < 0.0:
            #direction = -(90. + angle)
            corrected_angle = 180 + angle
        
        # Get opposite direction
        if corrected_angle <= 180:
            self.target_rotation = corrected_angle + 180
        else:
            self.target_rotation = corrected_angle - 180 
        
        #Move in a direction that is somewhat away from the chaser by sampling from a range
        low_theta = self.target_rotation - 60
        high_theta = self.target_rotation + 60
        sample_set = list(range(int(low_theta), int(high_theta)))
        sample_set = [a for a in sample_set if a <= 360 and a >= 0]
        ang = np.random.choice(sample_set).astype(float)
        
        # convert to radians and use trig to get new position
        opp_angle_rads = ang * np.pi / 180.
            
        if ang >= 0 and ang < 90:
            x_d_wall = self.x_limits[1] - runner_pos[0]
            z_d_wall = self.z_limits[1] - runner_pos[1]
        elif ang >= 90 and ang < 180:
            x_d_wall = self.x_limits[0] - runner_pos[0]
            z_d_wall = self.z_limits[1] - runner_pos[1]   
        elif ang >= 180 and ang < 270:
            x_d_wall = self.x_limits[0] - runner_pos[0]
            z_d_wall = self.z_limits[0] - runner_pos[1]
        elif ang >= 270 and ang < 361:
            x_d_wall = self.x_limits[0] - runner_pos[0]
            z_d_wall = self.z_limits[0] - runner_pos[1]
        # magnify size of unit circle we are drawing escape angle from based on minimum distance from wall
        closest_wall_dist = np.min(np.abs([x_d_wall, z_d_wall]))
        mag_factor = random.uniform(1, closest_wall_dist)
        x_step = (np.cos(opp_angle_rads) * mag_factor)
        z_step = (np.sin(opp_angle_rads) * mag_factor)
        assert np.allclose(np.tan(opp_angle_rads), z_step / x_step, atol=1e-4), 'Bad Angle'
        new_x = runner_pos[0] + x_step
        new_z = runner_pos[2] + z_step
        
        # if close to getting corner, get away from corner
        if np.abs(runner_pos[0]) > (self.x_limits[1] - 1.0) and np.abs(runner_pos[2]) > (self.z_limits[1] - 1.0):
            restricted_angles = list(range(int(corrected_angle)-30, int(corrected_angle)+30))
            which_corner = np.sign([runner_pos[0], runner_pos[1]])
            if which_corner[0] == -1 and which_corner[1] == -1:
                available_angles = [a for a in np.arange(0, 91) if a not in restricted_angles]
            elif which_corner[0] == -1 and which_corner[1] == 1:
                available_angles = [a for a in np.arange(270, 361) if a not in restricted_angles]
            elif which_corner[0] == 1 and which_corner[1] == -1:
                available_angles = [a for a in np.arange(90, 181) if a not in restricted_angles]
            elif which_corner[0] == 1 and which_corner[1] == 1:
                available_angles = [a for a in np.arange(180, 271) if a not in restricted_angles]
            ang = np.random.choice(available_angles).astype(float)
            # convert to radians and use trig to get new position
            opp_angle_rads = ang * np.pi / 180.
            if ang >= 0 and ang < 90:
                x_d_wall = self.x_limits[1] - runner_pos[0]
                z_d_wall = self.z_limits[1] - runner_pos[1]
            elif ang >= 90 and ang < 180:
                x_d_wall = self.x_limits[0] - runner_pos[0]
                z_d_wall = self.z_limits[1] - runner_pos[1]   
            elif ang >= 180 and ang < 270:
                x_d_wall = self.x_limits[0] - runner_pos[0]
                z_d_wall = self.z_limits[0] - runner_pos[1]
            elif ang >= 270 and ang < 361:
                x_d_wall = self.x_limits[0] - runner_pos[0]
                z_d_wall = self.z_limits[0] - runner_pos[1]
            # magnify size of unit circle we are drawing escape angle from based on minimum distance from wall
            closest_wall_dist = np.min(np.abs([x_d_wall, z_d_wall]))

            mag_factor = random.uniform(1, closest_wall_dist)
            x_step = (np.cos(opp_angle_rads) * mag_factor)
            z_step = (np.sin(opp_angle_rads) * mag_factor)
            assert np.allclose(np.tan(opp_angle_rads), z_step / x_step, atol=1e-4), 'Bad Angle'
            new_x = runner_pos[0] + x_step
            new_z = runner_pos[2] + z_step
        
        # make sure target location is in bounds
        if new_x > self.x_limits[1]:
            new_x = self.x_limits[1]
        elif new_x < self.x_limits[0]:
            new_x = self.x_limits[0]
        if new_z > self.z_limits[1]:
            new_z = self.z_limits[1]
        elif new_z < self.z_limits[0]:
            new_z = self.z_limits[0]
            
        self.new_runner_pos = [new_x, runner_pos[1], new_z]
        
    def act(self, obs):
        commands = []
        # repeat same transformation until avatar is close to destination
        if obs['avsb']:
            self.avatar_position = obs['avsb'][self.avatar_id]['transform']['position']
            # repeat same transformation until avatar is close to destination, then find new escape position
            if np.allclose(self.new_runner_pos, self.avatar_position, atol=1e-1):
                self.get_escape_angle(obs)
            
                if self.debug:
                    print("[runner]: RUNNING AWAY TO POS: {}".format(self.new_runner_pos))
                
        action = 'move_away' 
        commands.append({"$type": "move_avatar_towards_position", 
                         "position": {"x": self.new_runner_pos[0], "y": self.new_runner_pos[1], "z": self.new_runner_pos[2]}, 
                         "speed": self.step_size,
                         "avatar_id": self.avatar_id})
        
        if self.debug:
            print("[runner]: AT POS: {}".format(self.avatar_position))

        return action, commands
    
    
class MagnebotChasingAgent(PremotorAgent):
    def __init__(self, avatar_id, avatar, is_async, action_space, following_id):
        self.name = 'MagnebotChasingAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)       
        self.following_id = following_id
        self.counter = 0
        self.create_action_to_cmds_dict()
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def act(self, obs):
        if obs['avsb']:
            if isinstance(self.following_id, str):
                target_pos = np.array(obs['avsb'][self.following_id]['transform']['position'])
            elif isinstance(self.following_id, int):
                target_pos = np.array(obs['avsb'][self.following_id]['dynamic']['transform']['position'])
            
            action = "move"
            kwargs = {"target": target_pos}
            self.action_to_cmds_dict[action] = (self.avatar.move_to, kwargs)
            cmds = self.get_action_cmds(action)
        else:
            action = 'noop'
            cmds = []
            
        return action, cmds
    
    
class MagnebotRunnerAgent(PremotorAgent):
    def __init__(self, avatar_id, avatar, is_async, action_space, follower_id):
        self.name = 'MagnebotRunnerAgent'
        super().__init__(avatar_id, avatar, is_async, action_space)       
        self.follower_id = follower_id
        self.counter = 0
        self.create_action_to_cmds_dict()
        self.target_rotation = 0.
        self.debug = True
        self.avatar_position = np.array([0., 0., 0.])
        self.new_runner_pos = np.array([0., 0., 0.])
        # to do make this parametrized
        self.x_limits = (-5, 5)
        self.z_limits = (-5, 5)
        
    def create_action_to_cmds_dict(self):
        self.action_to_cmds_dict = {}
        
    def get_escape_angle(self, obs):
        if isinstance(self.follower_id, str):
            chaser_pos = obs['avsb'][self.follower_id]['transform']['position']
        elif isinstance(self.follower_id, int):
            chaser_pos = obs['avsb'][self.follower_id]['dynamic']['transform']['position']
        runner_pos = self.avatar.dynamic.transform.position
          
        x_d = chaser_pos[0] - runner_pos[0]
        z_d = chaser_pos[2] - runner_pos[2]
        
        # Getting the angle
        hyp_d = math.sqrt(x_d**2 + z_d**2)
        angle = math.degrees(np.arcsin(abs(z_d)/hyp_d)) # pythagoras; fyi: math.sin takes angle in radians
        
        # Direction that chaser is in
        if z_d >= 0.0 and x_d >= 0.0:
            #direction = 90. - angle
            corrected_angle = angle
        elif z_d >= 0.0 and x_d < 0.0:
            #direction = -(90. - angle)
            corrected_angle = 180 - angle
        elif z_d < 0.0 and x_d >= 0.0:
            #direction = 90. + angle
            corrected_angle = 360 - angle
        elif z_d < 0.0 and x_d < 0.0:
            #direction = -(90. + angle)
            corrected_angle = 180 + angle
        
        # Get opposite direction
        if corrected_angle <= 180:
            self.target_rotation = corrected_angle + 180
        else:
            self.target_rotation = corrected_angle - 180 
        
        #Move in a direction that is somewhat away from the chaser by sampling from a range
        low_theta = self.target_rotation - 60
        high_theta = self.target_rotation + 60
        sample_set = list(range(int(low_theta), int(high_theta)))
        sample_set = [a for a in sample_set if a <= 360 and a >= 0]
        ang = np.random.choice(sample_set).astype(float)
        
        # convert to radians and use trig to get new position
        opp_angle_rads = ang * np.pi / 180.
            
        if ang >= 0 and ang < 90:
            x_d_wall = self.x_limits[1] - runner_pos[0]
            z_d_wall = self.z_limits[1] - runner_pos[1]
        elif ang >= 90 and ang < 180:
            x_d_wall = self.x_limits[0] - runner_pos[0]
            z_d_wall = self.z_limits[1] - runner_pos[1]   
        elif ang >= 180 and ang < 270:
            x_d_wall = self.x_limits[0] - runner_pos[0]
            z_d_wall = self.z_limits[0] - runner_pos[1]
        elif ang >= 270 and ang < 361:
            x_d_wall = self.x_limits[0] - runner_pos[0]
            z_d_wall = self.z_limits[0] - runner_pos[1]
        # magnify size of unit circle we are drawing escape angle from based on minimum distance from wall
        closest_wall_dist = np.min(np.abs([x_d_wall, z_d_wall]))
        mag_factor = random.uniform(1, closest_wall_dist)
        x_step = (np.cos(opp_angle_rads) * mag_factor)
        z_step = (np.sin(opp_angle_rads) * mag_factor)
        assert np.allclose(np.tan(opp_angle_rads), z_step / x_step, atol=1e-4), 'Bad Angle'
        new_x = runner_pos[0] + x_step
        new_z = runner_pos[2] + z_step
        
        # if close to getting corner, get away from corner
        if np.abs(runner_pos[0]) > (self.x_limits[1] - 1.0) and np.abs(runner_pos[2]) > (self.z_limits[1] - 1.0):
            restricted_angles = list(range(int(corrected_angle)-30, int(corrected_angle)+30))
            which_corner = np.sign([runner_pos[0], runner_pos[1]])
            if which_corner[0] == -1 and which_corner[1] == -1:
                available_angles = [a for a in np.arange(0, 91) if a not in restricted_angles]
            elif which_corner[0] == -1 and which_corner[1] == 1:
                available_angles = [a for a in np.arange(270, 361) if a not in restricted_angles]
            elif which_corner[0] == 1 and which_corner[1] == -1:
                available_angles = [a for a in np.arange(90, 181) if a not in restricted_angles]
            elif which_corner[0] == 1 and which_corner[1] == 1:
                available_angles = [a for a in np.arange(180, 271) if a not in restricted_angles]
            ang = np.random.choice(available_angles).astype(float)
            # convert to radians and use trig to get new position
            opp_angle_rads = ang * np.pi / 180.
            if ang >= 0 and ang < 90:
                x_d_wall = self.x_limits[1] - runner_pos[0]
                z_d_wall = self.z_limits[1] - runner_pos[1]
            elif ang >= 90 and ang < 180:
                x_d_wall = self.x_limits[0] - runner_pos[0]
                z_d_wall = self.z_limits[1] - runner_pos[1]   
            elif ang >= 180 and ang < 270:
                x_d_wall = self.x_limits[0] - runner_pos[0]
                z_d_wall = self.z_limits[0] - runner_pos[1]
            elif ang >= 270 and ang < 361:
                x_d_wall = self.x_limits[0] - runner_pos[0]
                z_d_wall = self.z_limits[0] - runner_pos[1]
            # magnify size of unit circle we are drawing escape angle from based on minimum distance from wall
            closest_wall_dist = np.min(np.abs([x_d_wall, z_d_wall]))

            mag_factor = random.uniform(1, closest_wall_dist)
            x_step = (np.cos(opp_angle_rads) * mag_factor)
            z_step = (np.sin(opp_angle_rads) * mag_factor)
            assert np.allclose(np.tan(opp_angle_rads), z_step / x_step, atol=1e-4), 'Bad Angle'
            new_x = runner_pos[0] + x_step
            new_z = runner_pos[2] + z_step
        
        # make sure target location is in bounds
        if new_x > self.x_limits[1]:
            new_x = self.x_limits[1]
        elif new_x < self.x_limits[0]:
            new_x = self.x_limits[0]
        if new_z > self.z_limits[1]:
            new_z = self.z_limits[1]
        elif new_z < self.z_limits[0]:
            new_z = self.z_limits[0]
            
        self.new_runner_pos = np.array([new_x, runner_pos[1], new_z])
        
    def act(self, obs):
        # repeat same transformation until avatar is close to destination
        if obs['avsb']:
            self.avatar_position = self.avatar.dynamic.transform.position
            # repeat same transformation until avatar is close to destination, then find new escape position
            if np.allclose(self.new_runner_pos, self.avatar_position, atol=1e-1):
                self.get_escape_angle(obs)
            
                if self.debug:
                    print("[runner]: RUNNING AWAY TO POS: {}".format(self.new_runner_pos))
          
        action = "move_away"
        kwargs = {"target": self.new_runner_pos}
        self.action_to_cmds_dict[action] = (self.avatar.move_to, kwargs)
        cmds = self.get_action_cmds(action)
        
        if self.debug:
            print("[runner]: AT POS: {}".format(self.avatar_position))

        return action, cmds
        
            
if __name__ == '__main__':
    avatar_id = 1
    kwargs = {"position": {"x":0, "y":0, "z":0},
              "robot_id": avatar_id}
    avatar = Magnebot(**kwargs)
    action_space = []
    bot = DummyMagnebotAgent(avatar_id, avatar, False, action_space)
    while True:
        bot.act([])
    


