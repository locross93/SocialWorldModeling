import numpy as np
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from tdw.add_ons.avatar_body import AvatarBody
from magnebot.image_frequency import ImageFrequency

import os
import misc.utils as utils
import misc.commands as commands
import agents.agent_funcs as agent_funcs
import agents.agent as agent
from policy import ExistFirstPolicy
import itertools
from scipy.spatial import distance
import random
import datetime
    
save_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/swm_simulation_env/img_out/'

def generate_config(scenario_num=-1):
    scenario_set = [
        ['gathering', 'random'], # 0
        ['multistep', 'random'], # 1
        ['leader', 'follower'], # 2
        ['gathering', 'adversarial'], # 3
        ['random', 'random'], # 4
        ['random', 'mimic'], # 5
        ['runner', 'chaser'], # 6
        ['gathering', 'static'], # 7
        ['multistep', 'static'], # 8
        ]
    
    if scenario_num == -1:
        num_scenarios = len(scenario_set)
        scenario_num = np.random.randint(num_scenarios)
    behaviors = scenario_set[scenario_num]
    # randomly shuffle agent position in list
    flip = random.randint(0, 1)
    if flip == 0:
        behaviors.reverse()
    
    object_set = ['jug05', 'bronze_purse', 'bread', 'b04_honey_jar', 'backpack', 'vase_01']
    num_goals = 3
    object_list = random.sample(object_set, k=num_goals)
    
    # how many objects
    num_objects = len(object_list)
    
    # randomly sample object and agent positions
    # make sure entities are not close together with a spatial constraint
    num_actors = len(behaviors)
    num_locations = num_objects + num_actors
    constraints_met = False
    spatial_constraint = 1.0
    while not constraints_met:
        x_pos = np.random.uniform(-4, 4, num_locations)
        y_pos = np.zeros(num_locations)
        z_pos = np.random.uniform(-3, 5, num_locations)
        entity_positions = np.column_stack((x_pos, y_pos, z_pos))
        combos = list(itertools.combinations(entity_positions, 2))
        constraints_met = True
        for pair in combos:
            pair_dist = distance.euclidean(pair[0], pair[1])
            if pair_dist < spatial_constraint:
                constraints_met = False
        
    num_non_actors = num_locations - num_actors
    object_positions = [{'x': obj_pos[0], 'y': obj_pos[1], 'z': obj_pos[2]} for obj_pos in entity_positions[:num_non_actors,:]]
    agent_positions = [{'x': obj_pos[0], 'y': obj_pos[1], 'z': obj_pos[2]} for obj_pos in entity_positions[num_non_actors:,:]]
    
    #if collaborative gathering, make goal sequences for leader and follower
    if 'leader' in behaviors and 'follower' in behaviors:
        goal_perms = list(itertools.permutations(np.arange(num_goals)))
        N = len(goal_perms)
        agent_num = np.random.randint(N)
        agent_seq = goal_perms[agent_num]
        agent_seq_lead = [agent_seq[0], agent_seq[2]]
        agent_seq_follow = [agent_seq[1]]
    
    agent_cfgs = {}
    agent_func_dict = {}
    agent_ids = [3434, 1212]
    for i,behavior in enumerate(behaviors): 
        agent_id = agent_ids[i]
        other_agent_id = [temp_id for temp_id in agent_ids if temp_id != agent_id][0]
        agent_info = {
                'is_async': False,
                'action_space': ["move"],
                'position': agent_positions[i],
                'rotation': {"x": 0, "y": 0, "z": 0},
                'image_frequency': ImageFrequency.never}
        agent_func = {'cfg_func': agent_funcs.setup_magnebot_avatar}
        if behavior == 'gathering':
            agent_goal = np.random.randint(num_goals)
            agent_info['kwargs'] = {'target_obj': agent_goal, 'bring2observer': True}
            agent_func['init_func'] = agent.GatheringMagnebotAgent
        elif behavior == 'random':
            agent_info['kwargs'] = {'save_commands': True}
            agent_func['init_func'] = agent.RandomMovingMagnebotAgent
        elif behavior == 'multistep':
            goal_perms = list(itertools.permutations(np.arange(num_goals)))
            N = len(goal_perms)
            agent_num = np.random.randint(N)
            agent_seq = goal_perms[agent_num]
            agent_info['kwargs'] = {'goal_sequence': agent_seq, 'bring2observer': True}
            agent_func['init_func'] = agent.MultistepMagnebotAgent
        elif behavior == 'leader':
            agent_info['kwargs'] = {'goal_sequence': agent_seq_lead, 'bring2observer': True, 'collaborative': 'leader'}
            agent_func['init_func'] = agent.MultistepMagnebotAgent
        elif behavior == 'follower':
            agent_info['kwargs'] = {'goal_sequence': agent_seq_follow, 'bring2observer': True, 'collaborative': 'follower'}
            agent_func['init_func'] = agent.MultistepMagnebotAgent
        elif behavior == 'adversarial':
            agent_info['kwargs'] = {'obs2move': np.arange(num_goals), 'following_id': other_agent_id}
            agent_func['init_func'] = agent.AdversarialGatheringMagnebotAgentNoTable
        elif behavior == 'mimic':
            agent_info['kwargs'] = {'following_id': other_agent_id}
            agent_func['init_func'] = agent.MagnebotMimicAgent
        elif behavior == 'runner':
            agent_info['kwargs'] = {'follower_id': other_agent_id}
            agent_func['init_func'] = agent.MagnebotRunnerAgent
        elif behavior == 'chaser':
            agent_info['kwargs'] = {'following_id': other_agent_id}
            agent_func['init_func'] = agent.MagnebotChasingAgent
        agent_cfgs[agent_id] = agent_info
        agent_func_dict[agent_id] = agent_func
        
    date = datetime.datetime.now()
    date_str = str(date.month)+'-'+str(date.day)+'-'+str(date.year)+'-'+str(date.hour)+'-'+str(date.minute)
    save_folder = behaviors[0]+'_'+behaviors[1]+'_'+date_str+'/'
    
    agent_cfgs['fp1'] = {
        'is_async': False,
        'body': AvatarBody.cylinder,
        'action_space': ['r-left', 'r-right'],
        'position':{"x": 0, "y": 0.0, "z": -6.0},
        'look_at': {"x": 0, "y": 0.6, "z": 0},
        'rotation': {"x": 0, "y": 0, "z": 0},
        'dynamic_friction': 0.3,
        'static_friction': 0.3,
        'bounciness': 0.7,
        'mass': 40,
        'color': {"r": 0, "g": 1, "b": 0, "a": 0.9},
        'field_of_view': 70,
        'kwargs': {}}
    
    agent_func_dict['fp1'] = {
        'cfg_func': agent_funcs.setup_embodied_avatar,
        'init_func': agent.ObserverAgent}
    
    cfg = dict(
        basic_room_settings=[
            {"$type": "set_screen_size", "width": 512, "height": 512},
            TDWUtils.create_empty_room(15, 15),
            {"$type": "send_scene_regions"},
            {"$type": "bake_nav_mesh"}],
        model_library="models_core.json",
        num_objects=num_objects,
        object_list=object_list,
        object_positions=object_positions,
        behaviors=behaviors,
        # all the object data we want to save
        object_output_commands=[
            {"$type": "send_transforms", "frequency": "always"},
            {"$type": "send_rigidbodies", "frequency": "always"},
            {"$type": "make_nav_mesh_obstacle", "carve_type": "all", "scale": 1, "shape": "box"},
        ],
        img_settings={
            'save_img': True,
            'save_obs': True,
            'disable_cameras': True,
            'img_path': save_dir+save_folder
        },
        data_settings={
            'save_freqs': 1,    # save at every n steps
            'db_port': 27017,
            'db_name': 'Exp',
        },
        policy=ExistFirstPolicy,
        agent_cfgs=agent_cfgs,
        agent_funcs = agent_func_dict
    )
        
    return cfg
        



