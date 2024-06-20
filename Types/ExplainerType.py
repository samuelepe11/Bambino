# Import packages
from enum import Enum


# Class
class ExplainerType(Enum):
    GC = "GradCAM"
    HRC = "HiResCAM"
