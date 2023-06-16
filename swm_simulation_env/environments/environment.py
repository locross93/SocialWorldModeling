import os
import pdb
import numpy as np
from time import sleep

import tdw
from tdw.tdw_utils import TDWUtils
from tdw.controller import Controller
from tdw.add_ons.logger import Logger
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, SceneRegions, \
    Transforms, Rigidbodies, Images, NavMeshPath, IsOnNavMesh
from tdw.add_ons.step_physics import StepPhysics
# modules within this repo
import misc.utils as utils

tdw_lib_path = os.path.abspath(tdw.__file__).split('__init__.py')[0]
tdw_lib_path = os.path.join(tdw_lib_path, 'metadata_libraries')


class Environment(Controller):
    """Facilitates all the communications with TDW build.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        for k, v in cfg.items():
            setattr(self, k, v)
        super().__init__(port=self.build_port)
            
    def make_object_layout(self):
        """Put object in a room based on their configuration (objects&positions)
        """        
        assert len(self.object_positions) == len(self.object_list), \
            "Please offer enough initial object positions"
        self.obj_ids = []
        obj_commands = []        
        for i, obj in enumerate(self.object_list):
            obj_id = self.get_unique_id()
            position=self.object_positions[i]
            self.obj_ids.append(obj_id)
            obj_commands.append(
                self.get_add_object(
                    model_name=obj, object_id=obj_id, library=self.model_library,
                    position=position))
        added_mesh_cmds = False
        for cmd in self.object_output_commands:
            if cmd['$type'] == 'make_nav_mesh_obstacle' and not added_mesh_cmds:
                cmd['id'] = self.obj_ids[-1]
                # for id_num in self.obj_ids[1:]:
                #     id_mesh_cmd = cmd.copy()
                #     id_mesh_cmd['id'] = id_num
                #     self.object_output_commands.append(id_mesh_cmd)
                added_mesh_cmds = True
            elif cmd['$type'] != 'make_nav_mesh_obstacle':
                cmd['ids'] = self.obj_ids
        obj_commands.extend(self.object_output_commands)
        # add occupancy map
        self.occupancy_map = OccupancyMap(cell_size=0.5)
        self.add_ons.append(self.occupancy_map)
        resp = self.communicate(obj_commands)
        
        
    def make_agent_layout(self):
        """Instantiate all the avatars and put them in the environment. The avatars will later be used to create all agents in a room.
        """
        def _add_avatar_func_to_dict(avatar_id, avatar, avatar_funcs, avatar_dict):
            init_func = avatar_funcs['init_func']
            avatar_dict[avatar_id] = {'avatar': avatar,
                                      'init_func': init_func}            
        self.agents = {}
        self.async_agent_ids = []
        assert len(self.agent_cfgs) == len(self.agent_funcs), \
            "Number of cfgs doesn\'t equal to number of funcs"
        all_agent_ids = []
        for agent_id, cfg in self.agent_cfgs.items():
            print(agent_id)
            all_agent_ids.append(agent_id)
            if cfg['is_async']:
                self.async_agent_ids.append(agent_id)
            agent_funcs = self.agent_funcs[agent_id]
            cfg_func = agent_funcs['cfg_func']            
            avatar, cmds = cfg_func(agent_id, cfg)            
            _add_avatar_func_to_dict(agent_id, avatar, agent_funcs, self.agents)
            self.add_ons.append(avatar)
        self.capture = ImageCapture(avatar_ids=all_agent_ids, path=self.img_settings['img_path'])
        #self.capture.set(save=self.img_settings['save_img'])  
        self.capture.set(save=self.img_settings['save_img'], frequency="once")  
        #self.add_ons.append(self.capture)
        # step physics
        self.step_physics = StepPhysics(num_frames=20)
        #self.step_physics = StepPhysics(num_frames=1)
        # camera 
        # self.cam = ThirdPersonCamera(position={"x": 0, "y": 5.0, "z": -7.0},
        #                 look_at={"x": 0, "y": 0, "z": 0})
        # self.cam = ThirdPersonCamera(position={"x": 0, "y": 8.0, "z": -8.0},
        #                 look_at={"x": 0, "y": 0, "z": 0})
        # self.cam = ThirdPersonCamera(position={"x": 0, "y": 3.0, "z": -5.0}, 
        #                 look_at={'x': 2.5, 'y': 1.0, 'z': -1})
        self.cam = ThirdPersonCamera(position={"x": 0, "y": 3.0, "z": -5.0}, 
                        look_at=3434)
        # self.cam = ThirdPersonCamera(position={"x": 0, "y": 3.0, "z": 0.0}, 
        #                 look_at=6767)
        # self.cam = ThirdPersonCamera(position={"x": 0, "y": 1.0, "z": -6.0}, 
        #                 look_at=6767)
        # self.cam = ThirdPersonCamera(position={"x": -4, "y": 3.0, "z": -2.0},
        #                 look_at={"x": 0, "y": 0, "z": -2.0})
        if 'disable_cameras' in self.cfg['img_settings'] and self.cfg['img_settings']['disable_cameras']:
            #self.add_ons.extend([self.step_physics])
            self.add_ons.extend([self.capture, self.step_physics])
        else:
            self.add_ons.extend([self.capture, self.step_physics, self.cam])
            #self.add_ons.extend([self.capture, self.step_physics])
        
    def _process_agent_imgs(self):
        agent_imgs = {}
        agent_frames = self.capture.get_pil_images()
        for agent_id, agent_dict in agent_frames.items():
            pil_img = agent_dict['_img']
            agent_imgs[agent_id] = pil_img
        return agent_imgs
    
    def _get_nav_mesh_path(self, resp):
        nav_mesh_path = {}
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "path":
                nav_mesh = NavMeshPath(resp[i])
                nav_mesh_path = nav_mesh.get_path()
                nav_mesh_state = nav_mesh.get_state()
                if nav_mesh_state == "invalid":
                    nav_mesh_path = ["invalid"]
                break
            if r_id == 'isnm':
                isOnNavMesh = IsOnNavMesh(resp[i])
                
        return nav_mesh_path

    def _aggregate_output_data(self, resp):
        all_data = utils.process_tdw_output_data(resp)
        all_data['path'] = self._get_nav_mesh_path(resp)
        if 'disable_cameras' not in self.cfg['img_settings'] or self.cfg['img_settings']['disable_cameras'] is False:
            agent_imgs = self._process_agent_imgs()
            all_data['imag'] = agent_imgs
        return all_data
    
    def reset(self):
        if hasattr(self, "proc_gen_commands"):
            self.communicate(self.proc_gen_commands)
        else:
            self.communicate(self.basic_room_settings)
            self.make_object_layout()
        self.make_agent_layout()
        resp = self.communicate({"$type": "set_render_quality", "render_quality": 0})
        return self._aggregate_output_data(resp)
        
    def step(self, action_dict):
        """ Takes in action dict {avatar_id: action}, converts actions into TDW commands then update the environment with the actions
        """
        cmds = []
        for agent_id, cmd in action_dict.items():
            cmds.extend(cmd)
        resp = self.communicate(cmds)
        return self._aggregate_output_data(resp)
