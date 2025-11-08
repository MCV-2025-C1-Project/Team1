from .akaze import akaze_descriptor
from .sift import sift_descriptor
from .orb import orb_descriptor

from typing import Literal
import numpy as np

def generate_descriptor(img: np.ndarray, descriptor: Literal['sift', 'orb', 'color_sift'], **kwargs) -> tuple:
    if descriptor == 'sift':
        sift_keys = sift_descriptor.__code__.co_varnames[:sift_descriptor.__code__.co_argcount]
        sift_kwargs = {key: kwargs[key] for key in sift_keys if key in kwargs}
        return sift_descriptor(img, **sift_kwargs)
    elif descriptor == 'orb':
        orb_keys = orb_descriptor.__code__.co_varnames[:orb_descriptor.__code__.co_argcount]
        orb_kwargs = {key: kwargs[key] for key in orb_keys if key in kwargs}
        return orb_descriptor(img, **orb_kwargs)
    elif descriptor == 'akaze':
            akaze_keys = akaze_descriptor.__code__.co_varnames[:akaze_descriptor.__code__.co_argcount]
            akaze_kwargs = {key: kwargs[key] for key in akaze_keys if key in kwargs}
            return akaze_descriptor(img, **akaze_kwargs)