from typing import List
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from fishVR.core import FishVR, FishVRConfig, FishVRState
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

    def process(self, frame: NDArray, state: FishVRState) -> tuple:

        state = self.tail_tracker.track_tail(frame, state)  # track tail and update state                                               
        res, state = self.estimator.estimate(state)  # estimate strength and turning strength and update state

        return res, state
    
    def build_res(self):
        res = np.array(
            (
                'success': bool = True,
                tracking.body_axes, 
                tracking.body_axes_global,
                tracking.centroid_resized,
                tracking.centroid_cropped,
                tracking.centroid_input,
                tracking.centroid_global,
                tracking.angle_rad,
                tracking.angle_rad_global,
                mask, 
                preproc.image_processed,
                preproc.image_cropped,
                preproc.background_image_cropped,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
            ), 
            dtype=self.tracking_param.dtype
        )
        return res
    
    'centroid_global' - estimated x,y position
    'caudorostral_axis' - theta
    # mediolater, axis perpendicular to caudorostral axis