# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from DataUtils.OpenFaceInstance import OpenFaceInstance


# Class
class StimulusConv1d(nn.Module):

    def __init__(self, is_2d=False, params=None, n_classes=2, separated_inputs=True):
        super(StimulusConv1d, self).__init__()

        # Define attributes
        self.is_2d = is_2d
        self.params = params
        self.n_classes = n_classes
        self.separated_inputs = separated_inputs

        if params is None:
            self.layers = [64, 64, 64]
            self.kernel_size = 3
            self.hidden_dim = 64
            self.lr = 0.001
            self.batch_size = 32
            self.p_drop = 0.5
            self.n_extra_fc_after_conv = 0
            self.n_extra_fc_final = 0
        else:
            self.layers = [params["n_conv_neurons"]] * params["n_conv_layers"]
            self.kernel_size = params["kernel_size"]
            self.hidden_dim = params["hidden_dim"]
            self.lr = params["lr"]
            self.batch_size = params["batch_size"]
            self.p_drop = params["p_drop"]
            self.n_extra_fc_after_conv = params["n_extra_fc_after_conv"]
            self.n_extra_fc_final = params["n_extra_fc_final"]

        if separated_inputs:
            self.layer_dims = {"g": [OpenFaceInstance.dim_dict["g"]] + self.layers,
                               "h": [OpenFaceInstance.dim_dict["h"]] + self.layers,
                               "f": [OpenFaceInstance.dim_dict["f"]] + self.layers}
            self.blocks = list(OpenFaceInstance.dim_dict.keys())
            self.out_fc_dim = 3 * self.hidden_dim
        else:
            first_layer = [OpenFaceInstance.dim_dict["g"] + OpenFaceInstance.dim_dict["h"] +
                           OpenFaceInstance.dim_dict["f"]]
            self.layer_dims = {"a": first_layer + self.layers}
            self.blocks = ["a"]
            self.out_fc_dim = self.hidden_dim

        # Layers
        for block in self.blocks:
            for i in range(len(self.layer_dims[block]) - 1):
                setattr(self, "conv_" + block + str(i), nn.Conv1d(self.layer_dims[block][i], self.layer_dims[block][i + 1],
                                                                  kernel_size=self.kernel_size, stride=1))
                setattr(self, "relu_" + block + str(i), nn.ReLU())
                setattr(self, "pool_" + block + str(i), nn.MaxPool1d(kernel_size=2))
                setattr(self, "drop_" + block + str(i), nn.Dropout1d(p=self.p_drop))

            setattr(self, "fc_" + block + "0", nn.Linear(self.layer_dims[block][-1], self.hidden_dim))
            setattr(self, "relu_" + block + "0", nn.ReLU())
            for i in range(self.n_extra_fc_after_conv):
                setattr(self, "fc_" + block + str(i + 1), nn.Linear(self.hidden_dim, self.hidden_dim))
                setattr(self, "relu_" + block + str(i + 1), nn.ReLU())

        self.fc0 = nn.Linear(self.out_fc_dim, out_features=self.hidden_dim)
        self.relu0 = nn.ReLU()
        for i in range(self.n_extra_fc_final):
            setattr(self, "fc" + str(i + 1), nn.Linear(self.hidden_dim, self.hidden_dim))
            setattr(self, "relu" + str(i + 1), nn.ReLU())

        self.fc = nn.Linear(self.hidden_dim, out_features=self.n_classes)
        self.softmax = nn.Softmax(dim=1)

        # CAM attributes
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, layer_interrupt=None, age=None, trial_id=None):
        # Apply network
        target_activation = None
        outputs = []
        if not self.separated_inputs:
            x["a"] = torch.concat([x["g"], x["h"], x["f"]], dim=2)

        for block in self.blocks:
            out = x[block].permute(0, 2, 1)
            if not self.is_2d:
                dim = 2
            else:
                out = out.unsqueeze(1)
                dim = (2, 3)

            for i in range(len(self.layer_dims[block]) - 1):
                conv_layer = "conv_" + block + str(i)
                try:
                    out = getattr(self, conv_layer)(out)
                except RuntimeError:
                    padding = (self.kernel_size - 1) // 2
                    pad_dims = (padding, padding) if not self.is_2d else (padding, padding, padding, padding)
                    out = F.pad(out, pad_dims)
                    out = self.__dict__[conv_layer](out)
                if layer_interrupt == conv_layer:
                    target_activation = out
                    h = out.register_hook(self.activations_hook)

                out = getattr(self, "relu_" + block + str(i))(out)
                out = getattr(self, "pool_" + block + str(i))(out)
                out = getattr(self, "drop_" + block + str(i))(out)

            out = torch.mean(out, dim=dim)
            for i in range(self.n_extra_fc_after_conv + 1):
                out = getattr(self, "fc_" + block + str(i))(out)
                out = getattr(self, "relu_" + block + str(i))(out)
            outputs.append(out)
        out = torch.concat(outputs, dim=1)

        out = self.fc0(out)
        out = self.relu0(out)
        for i in range(self.n_extra_fc_final):
            out = getattr(self, "fc" + str(i + 1))(out)
            out = getattr(self, "relu" + str(i + 1))(out)

        if age is not None:
            device = getattr(self, "age_fc_0").bias.device
            if self.age_dim > 1:
                age = F.one_hot(age.to(dtype=torch.long), num_classes=self.age_dim)
            age = age.to(torch.float32).to(device)
            for i in range(len(self.age_fc_layers) - 1):
                age = getattr(self, "age_fc_" + str(i))(age)
                age = getattr(self, "age_relu_" + str(i))(age)
                age = getattr(self, "age_drop_" + str(i))(age)
            out = torch.cat([out, age], dim=1)
        if trial_id is not None:
            device = getattr(self, "trial_fc_0").bias.device
            if self.trial_dim > 1:
                trial_id = F.one_hot(trial_id.to(dtype=torch.long), num_classes=self.trial_dim)
            trial_id = trial_id.to(torch.float32).to(device)
            for i in range(len(self.trial_fc_layers) - 1):
                trial_id = getattr(self, "trial_fc_" + str(i))(trial_id)
                trial_id = getattr(self, "trial_relu_" + str(i))(trial_id)
                trial_id = getattr(self, "trial_drop_" + str(i))(trial_id)
            out = torch.cat([out, trial_id], dim=1)

        out = self.fc(out)

        if layer_interrupt is not None:
            return out, target_activation
        out = self.softmax(out)
        return out

    def set_training(self, training=True):
        if training:
            self.train()
        else:
            self.eval()

        # Set specific layers
        for layer in self.__dict__.keys():
            if isinstance(self.__dict__[layer], nn.Module):
                # Set training/eval mode per each interested layer
                if "drop" in layer or "batch_norm" in layer:
                    self.__dict__[layer].training = training

    def set_cuda(self, cuda=True):
        if cuda:
            self.cuda()
        else:
            self.cpu()

        # Set specific layers
        for layer in self.__dict__.keys():
            if isinstance(self.__dict__[layer], nn.Module):
                # Set cuda devise for parallelization
                if cuda:
                    self.__dict__[layer].cuda()
                else:
                    self.__dict__[layer].cpu()
