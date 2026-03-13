from typing import List
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from head_embedded.core import HeadEmbeddedConfig, HeadEmbeddedState, HeadEmbedded
from head_embedded.utils.tail_tracker import TailTracker
from head_embedded.utils.estimator import Estimator

@dataclass
class SingleFishHeadEmbeddedConfig(HeadEmbeddedConfig):
    
    @property
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('success', np.bool_),
            ('body_axes_global', np.float32, (2,2)),
            ('centroid_global', np.float32, (2,)),
            ('est_theta', np.float32, (1,)),
            ('strength', np.float32, (1,)),
            ('turning_strength', np.float32, (1,)),
            ('v_feedback_pix', np.float32, (1,)),
            ('omega_feedback_rad', np.float32, (1,)),
            ('tail_points_transformed', np.float32, (self.n_segments+1, 2)),
        ])
        return dt

    @property
    def failed(self):
        return np.zeros((), dtype=self.dtype)

@dataclass
class DummyClass:
    
    @property
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('success', np.bool_),
            ('body', NDArray),
        ])
        return dt

    @property
    def failed(self):
        return np.zeros((), dtype=self.dtype)
    
  
class SingleFishHeadEmbedded(HeadEmbedded):

    def __init__(
            self, 
            config: SingleFishHeadEmbeddedConfig = SingleFishHeadEmbeddedConfig(),
            tail_tracker: TailTracker = TailTracker(),
            estimator: Estimator = Estimator(),
        ):

        self.config = config
        self.tail_tracker = tail_tracker
        self.estimator = estimator

    def process(self, frame: NDArray, state: HeadEmbeddedState) -> tuple:

        state = self.tail_tracker.track_tail(frame, state)  # track tail and update state                                               
        state = self.estimator.estimate(state)  # estimate strength and turning strength and update state
        res = self.build_res(state)  # build result array from state
        estimator = np.array(
            (
                True, 
                res,
            ), dtype=DummyClass().dtype
        )

        return estimator, state
    
    def build_res(self, state):

        res = np.array(
            (
                True,
                # TODO: check this, currently 90 is top, 0 is right
                np.array([
                    [np.cos(state.theta), np.sin(state.theta)],
                    [np.sin(state.theta), -np.cos(state.theta)]
                ]), 
                np.array([state.x, state.y]),
                np.array([state.theta]),
                np.array([state.strength]),
                np.array([state.turning_strength]),
                np.array([state.v_feedback_pix]),
                np.array([state.omega_feedback_rad]),
                np.array(state.tail_points_transformed),
            ), 
            dtype=self.config.dtype
        )
        return res