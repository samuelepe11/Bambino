# Import packages
import torch.nn as nn

from DataUtils.OpenFaceInstance import OpenFaceInstance
from Networks.StimulusConv1d import StimulusConv1d


# Class
class StimulusConv2d(StimulusConv1d):

    def __init__(self, params=None, n_classes=2, separated_inputs=True):
        super(StimulusConv2d, self).__init__(is_2d=True, params=params, n_classes=n_classes,
                                             separated_inputs=separated_inputs)

        # Convert the 1D network into the 2D version
        StimulusConv2d.make_2d(model_object=self, params=params, separated_inputs=separated_inputs)

    @staticmethod
    def make_2d(model_object, params, separated_inputs):
        if params is None:
            model_object.layers = [64, 64, 64]
            model_object.kernel_size = 3
            model_object.hidden_dim = 64
            model_object.lr = 0.001
            model_object.p_drop = 0.5
            model_object.n_extra_fc_after_conv = 0
            model_object.n_extra_fc_final = 0

        layer_dims = [1] + model_object.layers
        if separated_inputs:
            model_object.layer_dims = {"g": layer_dims,
                                       "h": layer_dims,
                                       "f": layer_dims}
        else:
            model_object.layer_dims = {"a": layer_dims}

        # Layers
        for block in model_object.blocks:
            for i in range(len(model_object.layer_dims[block]) - 1):
                setattr(model_object, "conv_" + block + str(i), nn.Conv2d(model_object.layer_dims[block][i],
                                                                          model_object.layer_dims[block][i + 1],
                                                                          kernel_size=model_object.kernel_size,
                                                                          stride=1))
                setattr(model_object, "relu_" + block + str(i), nn.ReLU())
                setattr(model_object, "pool_" + block + str(i), nn.MaxPool2d(kernel_size=(1, 2)))
                setattr(model_object, "drop_" + block + str(i), nn.Dropout2d(p=model_object.p_drop))

            setattr(model_object, "fc_" + block + "0", nn.Linear(model_object.layer_dims[block][-1], model_object.hidden_dim))
            setattr(model_object, "relu_" + block + "0", nn.ReLU())
            for i in range(model_object.n_extra_fc_after_conv):
                setattr(model_object, "fc_" + block + str(i + 1), nn.Linear(model_object.hidden_dim, model_object.hidden_dim))
                setattr(model_object, "relu_" + block + str(i + 1), nn.ReLU())

        model_object.fc0 = nn.Linear(model_object.out_fc_dim, out_features=model_object.hidden_dim)
        model_object.relu0 = nn.ReLU()
        for i in range(model_object.n_extra_fc_final):
            setattr(model_object, "fc" + str(i + 1), nn.Linear(model_object.hidden_dim, model_object.hidden_dim))
            setattr(model_object, "relu" + str(i + 1), nn.ReLU())
