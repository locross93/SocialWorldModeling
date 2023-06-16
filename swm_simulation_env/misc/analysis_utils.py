import csv
import pandas as pd
import numpy as np
import pdb

def save_traj_observations(trial_obs, config, save_dir):
    num_frames = len(trial_obs)
    obs_dict = {}
    last_obs = trial_obs[-1]
    obj_keys = last_obs['tran'].keys()
    for obj_num,key in enumerate(obj_keys):
        temp_obj = config['object_list'][obj_num]
        obs_dict['obj'+str(obj_num)+'_name'] = [temp_obj for i in range(num_frames)]
        obs_dict['obj'+str(obj_num)+'_x'] = []
        obs_dict['obj'+str(obj_num)+'_y'] = []
        obs_dict['obj'+str(obj_num)+'_z'] = []
        obs_dict['obj'+str(obj_num)+'_rot_x'] = []
        obs_dict['obj'+str(obj_num)+'_rot_y'] = []
        obs_dict['obj'+str(obj_num)+'_rot_z'] = []
        obs_dict['obj'+str(obj_num)+'_rot_w'] = []
        for temp_obs in trial_obs:
            temp_pos = temp_obs['tran'][key]
            obs_dict['obj'+str(obj_num)+'_x'].append(temp_pos[0])
            obs_dict['obj'+str(obj_num)+'_y'].append(temp_pos[1])
            obs_dict['obj'+str(obj_num)+'_z'].append(temp_pos[2])
            temp_rot = temp_obs['rotation'][key]
            obs_dict['obj'+str(obj_num)+'_rot_x'].append(temp_rot[0])
            obs_dict['obj'+str(obj_num)+'_rot_y'].append(temp_rot[1])
            obs_dict['obj'+str(obj_num)+'_rot_z'].append(temp_rot[2])
            obs_dict['obj'+str(obj_num)+'_rot_w'].append(temp_rot[3])
            
    agent_keys = last_obs['avsb'].keys()
    for agent_num,key in enumerate(agent_keys):
        if isinstance(key, int):
            temp_agent = 'magnebot'
        else:
            temp_agent = config['agent_cfgs'][key]['body']._name_
        obs_dict['agent'+str(agent_num)+'_name'] = [temp_agent for i in range(num_frames)]
        obs_dict['agent'+str(agent_num)+'_x'] = []
        obs_dict['agent'+str(agent_num)+'_y'] = []
        obs_dict['agent'+str(agent_num)+'_z'] = []
        obs_dict['agent'+str(agent_num)+'_rot_x'] = []
        obs_dict['agent'+str(agent_num)+'_rot_y'] = []
        obs_dict['agent'+str(agent_num)+'_rot_z'] = []
        obs_dict['agent'+str(agent_num)+'_rot_w'] = []
        for temp_obs in trial_obs:
            if temp_agent == 'magnebot':
                temp_pos = np.array(temp_obs['avsb'][key]['dynamic']['transform']['position'])
                temp_rot = np.array(temp_obs['avsb'][key]['dynamic']['transform']['rotation'])
            else:
                temp_pos = np.array(temp_obs['avsb'][key]['transform']['position'])
                temp_rot = np.array(temp_obs['avsb'][key]['transform']['rotation'])
            obs_dict['agent'+str(agent_num)+'_x'].append(temp_pos[0])
            obs_dict['agent'+str(agent_num)+'_y'].append(temp_pos[1])
            obs_dict['agent'+str(agent_num)+'_z'].append(temp_pos[2])
            obs_dict['agent'+str(agent_num)+'_rot_x'].append(temp_rot[0])
            obs_dict['agent'+str(agent_num)+'_rot_y'].append(temp_rot[1])
            obs_dict['agent'+str(agent_num)+'_rot_z'].append(temp_rot[2])
            obs_dict['agent'+str(agent_num)+'_rot_w'].append(temp_rot[3])
        
    obs_df = pd.DataFrame(obs_dict)
    obs_df.to_csv(save_dir+'trial_obs.csv')
    
    #create an empty list to store the event log data
    event_log = []
    #loop through the event log list
    if 'comm' in last_obs:
        for item in last_obs['comm']:
            #check if the item is a string and starts with 'agent'
            if isinstance(item, str) and item.startswith('agent'):
                #split the item into three parts based on spaces
                agent, event, frame = item.split(' ')
                #remove the 'agent_' prefix from the agent part
                agent = agent.replace('agent_', '')
                #remove the 'frame_' prefix from the frame part
                frame = frame.replace('frame_', '')
                #append the parts as a tuple to the event log list
                event_log.append((agent, event, frame))
    #create a dataframe from the event log list with the column names
    event_df = pd.DataFrame(event_log, columns=['Agent', 'Event', 'Frame'])
    #write the dataframe to a csv file with the column names
    event_df.to_csv(save_dir+'event_log.csv', index=False)
