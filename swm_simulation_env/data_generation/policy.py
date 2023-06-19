import pdb
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from room import Room

class Policy:
    """Base class for user-avatar-room assignment in multi-agent setting.    

    This class allows experimenters to decide how they want to allow users to
    join rooms in a RoomQueue.
    """
    def __init__(self, cfg, rooms, current_room_id, room_user_assignment, **kwargs):
        self.cfg = cfg
        self.rooms = rooms
        self.current_room_id = current_room_id
        self.room_user_assignment = room_user_assignment
        self.kwargs = kwargs
        
    @abstractmethod
    def make_assignment(self):
        pass

    def update_room_user_assignment(self, room_user_assignment):
        """Update room_user_assignment when changes are made"""
        self.room_user_assignment = room_user_assignment
        self.make_new_decisions_upon_leaving()
        
    def make_new_decisions_upon_leaving(self):
        """A method to allow experimenter to decide how to handle user leaving in the middle of the experiment"""
        raise NotImplemented
        
class ExistFirstPolicy(Policy):
    """When a new user joins, assign them to existing room with free agents, if such room doesn't exist, create a new room.
    """
    def __init__(self, cfg, rooms, current_room_id, room_user_assignment, **kwargs):
        super().__init__(cfg, rooms, current_room_id, room_user_assignment, **kwargs)
        self.name = "ExistingFirstPolicy"
        
    def find_available_room(self):
        for room_id, room in self.rooms.items():
            if room_id is not None and len(room.free_agent_ids) > 0:
                return room_id
        return False

    def create_new_room(self):
        room = Room(self.current_room_id, self.cfg, **self.kwargs)
        self.rooms[self.current_room_id] = room
        self.current_room_id += 1
        return self.current_room_id - 1
            
    def make_assignment(self, user_id):
        room_id = self.find_available_room()
        if not room_id:
            room_id = self.create_new_room()            
        room = self.rooms[room_id]
        print(user_id, room_id)
        return room_id, room.free_agent_ids.pop()
