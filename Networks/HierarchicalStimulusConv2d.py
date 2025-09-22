# Import packages
import torch.nn as nn

from Networks.HierarchicalStimulusConv1d import HierarchicalStimulusConv1d
from Networks.StimulusConv2d import StimulusConv2d


# Class
class HierarchicalStimulusConv2d(HierarchicalStimulusConv1d):

    def __init__(self, age_dim, trial_dim, params=None, n_classes=2, separated_inputs=True):
        super(HierarchicalStimulusConv2d, self).__init__(age_dim=age_dim, trial_dim=trial_dim, params=params,
                                                         n_classes=n_classes, separated_inputs=separated_inputs,
                                                         is_2d=True)
        # Convert the 1D network into the 2D version
        StimulusConv2d.make_2d(model_object=self, params=params, separated_inputs=separated_inputs)
