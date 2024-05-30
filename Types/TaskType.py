# Import packages
from enum import Enum


# Class
class TaskType(Enum):
    STIM = "discriminate between stimulus and control"
    AGE = "learn the age of the subject"
    TRIAL = "learn the trial phase"
