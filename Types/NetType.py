# Import packages
from enum import Enum


# Class
class NetType(Enum):
    CONV1D = "convolutional 1D"
    CONV2D = "convolutional 2D"
    H_CONV1D = "hierarchical convolutional 1D"
    H_CONV2D = "hierarchical convolutional 2D"
