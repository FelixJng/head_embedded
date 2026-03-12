from typing import List
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from fishVR.core import FishVR, HeadEmbeddConfig, FishVRState
from fishVR.utils.tail_tracker import TailTracker
from fishVR.utils.estimator import Estimator

@dataclass
class SingleFishVRConfig(HeadEmbeddConfig):
    
    @property
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('success', np.bool_),
            ('body_axes', np.float32, (2,2)),
            ('body_axes_global', np.float32, (2,2)),
            ('centroid_resized', np.float32, (2,)),
            ('centroid_cropped', np.float32, (2,)),
            ('centroid_input', np.float32, (2,)),
            ('centroid_global', np.float32, (2,)),
            ('angle_rad', np.float32),
            ('angle_rad_global', np.float32),
            # ('mask', np.bool_, self.resized_dimension_px[::-1]),
            # ('image_processed', np.float32, self.resized_dimension_px[::-1]),
            # ('image_cropped', self.input_image_dtype, self.crop_dimension_px[::-1]),
            # ('background_image_cropped', self.input_image_dtype, self.crop_dimension_px[::-1]), 
            # ('pix_per_mm_global', np.float32),
            # ('pix_per_mm_input', np.float32),
            # ('pix_per_mm_cropped', np.float32),
            # ('pix_per_mm_resized', np.float32),
        ])
        return dt

    @property
    def failed(self):
        return np.zeros((), dtype=self.dtype)


  
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
                True,
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
            dtype=self.config.dtype
        )
        return res
    
    'centroid_global' - estimated x,y position
    'caudorostral_axis' - theta
    # mediolater, axis perpendicular to caudorostral axis