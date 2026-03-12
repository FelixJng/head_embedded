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