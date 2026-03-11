from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from numba import njit


@dataclass
class FishVRConfig:
    """Configuration for the FishVR system."""

    # General settings - camera
    acquisition_rate: int = 120

    # General settings - projector
    refresh_rate: int = 240
    parity_projector: int = 1
    dt: float = field(init=False)

    # General params - tail tracking
    n_segments: int = 20
    tail_length_pix: int = 120
    R: NDArray = np.array([[1, 0], [0,  1],])
    initial_pos: NDArray = np.array([960, 540])
    window_radius: int = 6
    initial_delta: NDArray = field(init=False)
    segment_length: int = field(init=False)
    kernel_size: int = 3
    n_interations: int = 1
    scalar_product_threshold: float = 0.7

    # General params - estimator
    window_width: float = 0.100  # in seconds, width of the window used for the estimator of v and omega
    tau_lp_now: float = 0.0
    alpha: float = 0.2
    beta: float = 175.0
    cm_per_pix: float = 1/404.7
    
    def __post_init__(self):
        self.dt = 1 / self.acquisition_rate
        self.initial_delta = np.array([0, self.tail_length_pix/self.n_segments])
        self.segment_length = self.tail_length_pix // self.n_segments

    # resolution: Tuple[int, int] = (1920, 1080)
    # color_mode: str = "BGR"  # or "RGB"


@dataclass
class FishVRState:
    """
    State of the FishVR system. For passing around chunk data.
    """
    def __init__(self, config: FishVRConfig = FishVRConfig()):
        self.config = config
    strength: float = 0.0
    turning_strength: float = 0.0
    strengths: List = []
    turning_strengths: List = []

    tail_points: NDArray = field(init=False)
    tail_points_list: List = []

    ww: int = field(init=False)
    last_values_1: List = field(init=False)
    last_values_2: List = field(init=False)

    # todo omega list etc?


    def __post_init__(self):
        self.tail_points = np.full((self.config.n_segments+1, 2), np.nan)
        self.tail_points[0] = self.config.initial_pos
        self.ww = int(np.ceil(self.config.window_width*self.config.acquisition_rate))
        self.last_values_1 = list(np.zeros(self.ww))
        self.last_values_2 = list(np.zeros(self.ww))

# class FishVRStateHandler:
#     """Handler for the FishVRState, responsible for updating the state based on new data."""
#     def __init__(self, state: FishVRState = FishVRState()):
#         self.state = state

#     def return_tail_tracking(self):


#     def update_state(self, new_data: Dict[str, Any]):
#         # Update the state based on new data
#         pass


class FishVR(ABC):
    """Abstract base class for the FishVR system."""

    # def __init__(self, config: FishVRConfig):
    #     self.config = config
    @abstractmethod
    def process(self, frame: NDArray) -> Dict[str, Any]:
        """Process a single frame and return relevant data."""
        pass
