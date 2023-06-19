""" dummy config to test data flow of the pipeline """
import numpy as np

from tdw.tdw_utils import TDWUtils
from tdw.add_ons.embodied_avatar import EmbodiedAvatar
from tdw.add_ons.first_person_avatar import FirstPersonAvatar
from magnebot import Magnebot
from magnebot.image_frequency import ImageFrequency

import misc.commands as commands


def setup_embodied_avatar(avatar_id, cfg):
    agent = EmbodiedAvatar(
        avatar_id=avatar_id,
        body=cfg['body'],
        position=cfg['position'],
        rotation=cfg['rotation'],        
        dynamic_friction=cfg['dynamic_friction'],        
        static_friction=cfg['static_friction'],
        bounciness=cfg['bounciness'],
        mass=cfg['mass'],
        color=cfg['color'],
        field_of_view=cfg['field_of_view'])
    cmds = [
        {"$type": "send_transforms",
         "frequency": "always",
         "ids": [avatar_id]},
        {"$type": "send_rigidbodies",
         "frequency": "always",
         "ids": [avatar_id]}
    ]
    return agent, cmds


def setup_first_person_avatar(avatar_id, cfg):
    agent = FirstPersonAvatar(
        avatar_id=avatar_id,
        position=cfg['position'],
        field_of_view=cfg['field_of_view'],
        framerate=cfg['framerate'],
        rotation=cfg['rotation']
    )
    cmds = []
    return agent, cmds


def setup_magnebot_avatar(avatar_id, cfg):
    if 'image_frequency' in cfg:
        image_frequency = cfg['image_frequency']
    else:
        image_frequency = ImageFrequency.always
    agent = Magnebot(
        robot_id=avatar_id,
        position=cfg['position'],
        rotation=cfg['rotation'],
        image_frequency=image_frequency
    )
    cmds = []
    return agent, cmds
