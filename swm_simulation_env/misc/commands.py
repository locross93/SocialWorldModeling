"""
Functions that return a list of commands to control the TDWBase environment
"""

from tdw.tdw_utils import TDWUtils
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen


def get_proc_gen_cmds(rng):
    proc_gen_kitchen = ProcGenKitchen()
    proc_gen_kitchen.create(rng=rng)
    cmds = proc_gen_kitchen.commands
    object_ids = [cmd['id'] for cmd in cmds if 'add_object' in cmd.values()]
    cmds.extend([
	{"$type": "send_transforms", "frequency": "always", "ids": object_ids},
        {"$type": "send_rigidbodies", "frequency": "always", "ids": object_ids}
    ])
    return cmds
