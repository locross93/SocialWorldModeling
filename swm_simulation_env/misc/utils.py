import json
import inspect
import numpy as np
from PIL.JpegImagePlugin import JpegImageFile
from tdw.librarian import ModelLibrarian
from tdw.add_ons.avatar_body import AvatarBody
from tdw.output_data import OutputData, Transforms, Rigidbodies, \
    SceneRegions


class MongoEncoder(json.JSONEncoder):
    """Used to convert foreign datatype to the ones mongo knows.
    
    A lot of classes, methods and TDW objects will be changed to
    their name to indicate what was used in the experiment.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif inspect.isclass(obj):
            return obj.__dict__['__module__']
        elif isinstance(obj, AvatarBody):
            return obj.__dict__['_name_']
        elif callable(obj):
            return f"func: {obj.__name__}"
        elif isinstance(obj, JpegImageFile):
            return []
        else:
            return super(MongoEncoder, self).default(obj)
        

def get_all_objects():
    """A helper function to get all the objects in the model library"""
    lib = ModelLibrarian()
    obj_list = [record.name for record in lib.records]
    return obj_list
    

def process_tdw_output_data(resp):
    """Convert TDW byte output to dict format. This is used for getting object data

    Arguments:
    resp -- response returned from controller.communicate() call
    """
    all_data = {}
    # ignore last output -- frame number
    for i in range(len(resp) - 1):    
        data = {}
        r_id = OutputData.get_data_type_id(resp[i])        
        if r_id == 'sreg':
            for j in range(scene_regions.get_num()):                
                data[j] = scene_regions.get_bounds(j)
        elif r_id == 'tran':
            all_data['rotation'] = {}
            transforms = Transforms(resp[i])
            for j in range(transforms.get_num()):
                obj_id = transforms.get_id(j)
                obj_pos = transforms.get_position(j)
                data[obj_id] = obj_pos
                obj_rot = transforms.get_rotation(j)
                all_data['rotation'][obj_id] = obj_rot
        elif r_id == 'rigi':
            rigidbodies = Rigidbodies(resp[i])
            for j in range(rigidbodies.get_num()):
                obj_id = rigidbodies.get_id(j)
                sleeping = rigidbodies.get_sleeping(j)
                data[obj_id] = sleeping
        all_data[r_id] = data
    return all_data
