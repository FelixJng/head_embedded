from typing import List
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from fishVR.core import FishVR, FishVRConfig
from fishVR.utils.tail_tracker import TailTracker
from fishVR.utils.estimator import Estimator

@dataclass
class SingleFishVRConfig(FishVRConfig):
    

    # Initialize tail tracking data (for chunk) 
    # this is only temporary 
    strength: float = 0.0
    turning_strength: float = 0.0
    strengths: List = []
    turning_strengths: List = []

    tail_points: NDArray = field(init=False)
    tail_points_list: List = []

    def __post_init__(self):
        super().__post_init__()
        self.tail_points = np.full((self.n_segments+1, 2), np.nan)
        self.tail_points[0] = self.initial_pos



  
class SingleFishVR(FishVR):

    def __init__(
            self, 
            config: SingleFishVRConfig = SingleFishVRConfig(),
            tail_tracker: TailTracker = TailTracker(),
            estimator: Estimator = Estimator()
        ):

        self.config = config
        self.tail_tracker = tail_tracker
        self.estimator = estimator
    