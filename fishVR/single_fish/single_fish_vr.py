from typing import List
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from fishVR.core import FishVR, FishVRConfig
from fishVR.utils.tail_tracker import TailTracker
from fishVR.utils.estimator import Estimator

@dataclass
class SingleFishVRConfig(FishVRConfig):
    pass
  
class SingleFishVR(FishVR):

    def __init__(
            self, 
            config: SingleFishVRConfig = SingleFishVRConfig(),
            tail_tracker: TailTracker = TailTracker(),
            estimator: Estimator = Estimator(),
        ):

        self.config = config
        self.tail_tracker = tail_tracker
        self.estimator = estimator

    def process(self, frame, state):
        
        tail_points_list = self.state_handler.state.tail_points_list
        tail_points = self.state_handler.state.tail_points

        # if not first_frame:
        #     pass

        # todo: i > 1!!!
        tail_points_transformed = self.tail_tracker.track_tail(frame, tail_points)  # track tail and update state
        tail_points_list.append(np.array(tail_points_transformed))

        
        force_lighthill,torque_lighthill=calculate_lighthill_force_torque(tail_points_list, vr_dict['acquisition_rate']) 

        return res