# Import packages
import torch.nn as nn

from Networks.StimulusConv1d import StimulusConv1d


# Class
class HierarchicalStimulusConv1d(StimulusConv1d):

    def __init__(self, age_dim, trial_dim, params=None, n_classes=2, separated_inputs=True):
        super(HierarchicalStimulusConv1d, self).__init__(is_2d=False, params=params, n_classes=n_classes,
                                                         separated_inputs=separated_inputs)

        # Define attributes
        self.age_dim = age_dim
        self.trial_dim = trial_dim
        if params is None:
            self.n_age_fc_neurons = 2
            self.age_fc_layers = [self.n_age_fc_neurons]
            self.n_trial_fc_neurons = 2
            self.trial_fc_layers = [self.n_trial_fc_neurons]
        else:
            self.n_age_fc_neurons = params["n_age_fc_neurons"]
            self.age_fc_layers = [self.n_age_fc_neurons] * params["n_age_fc_layers"]
            self.n_trial_fc_neurons = params["n_trial_fc_neurons"]
            self.trial_fc_layers = [self.n_trial_fc_neurons] * params["n_trial_fc_layers"]
        self.age_fc_layers = [age_dim] + self.age_fc_layers
        self.trial_fc_layers = [trial_dim] + self.trial_fc_layers

        # Layers
        for i in range(len(self.age_fc_layers) - 1):
            setattr(self, "age_fc_" + str(i), nn.Linear(self.age_fc_layers[i], out_features=self.age_fc_layers[i + 1]))
            setattr(self, "age_relu_" + str(i), nn.ReLU())
            setattr(self, "age_drop_" + str(i), nn.Dropout1d(p=self.p_drop))

        for i in range(len(self.trial_fc_layers) - 1):
            setattr(self, "trial_fc_" + str(i), nn.Linear(self.trial_fc_layers[i], out_features=self.trial_fc_layers[i + 1]))
            setattr(self, "trial_relu_" + str(i), nn.ReLU())
            setattr(self, "trial_drop_" + str(i), nn.Dropout1d(p=self.p_drop))

        self.fc = nn.Linear(self.hidden_dim + self.n_age_fc_neurons + self.n_trial_fc_neurons, out_features=self.n_classes)

    def set_training(self, training=True):
        if training:
            self.train()
        else:
            self.eval()

    def set_cuda(self, cuda=True):
        if cuda:
            self.cuda()
        else:
            self.cpu()
