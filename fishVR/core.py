from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
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

    # General params - tail tracking
    n_segments: int = 20
    tail_length_pix: int = 120
    R: np.array = np.array([
        [1, 0],
        [0,  1]
        ]),
    initial_pos: np.array = np.array([960, 540])
    window_radius: int = 6
    initial_delta: np.array = field(init=False)
    segment_length: int = field(init=False)
    kernel_size: int = 3
    n_interations: int = 1

    # General params - estimator
    window_width: float = 0.100  # in seconds, width of the window used for the estimator of v and omega
    tau_lp_now: float = 0.0
    alpha: float = 0.2
    beta: float = 175.0
    
    def __post_init__(self):
        self.initial_delta = np.array([0, self.tail_length_pix/self.n_segments])
        self.segment_length = self.tail_length_pix // self.n_segments

    # resolution: Tuple[int, int] = (1920, 1080)
    # color_mode: str = "BGR"  # or "RGB"




class FishVR(ABC):
    """Abstract base class for the FishVR system."""

    def __init__(self, config: FishVRConfig):
        self.config = config

    @abstractmethod
    def process_cropped_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame and return relevant data."""
        pass
