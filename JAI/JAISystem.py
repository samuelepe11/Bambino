# Import packages
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.OpenFaceInstance import OpenFaceInstance
from TrainUtils.NetworkTrainer import NetworkTrainer
from Types.ExplainerType import ExplainerType


# Class
class JAISystem:
    def __init__(self, working_dir, model_name):
        # Initialize attributes
        self.working_dir = working_dir
        self.results_dir = working_dir + OpenFaceDataset.results_fold
        self.models_dir = self.results_dir + OpenFaceDataset.models_fold
        self.jai_dir = self.results_dir + OpenFaceDataset.jai_fold

        self.model_name = model_name
        if model_name not in os.listdir(self.jai_dir):
            os.mkdir(self.jai_dir + model_name)

        self.trainer = NetworkTrainer.load_model(working_dir=working_dir, model_name=model_name)
        self.trainer.show_model()

    def get_cam(self, data_descr, target_layer_id, target_class, explainer_type, show=False):
        x, y = self.get_item(data_descr)
        cams, output_prob = self.draw_cam(self.trainer, x, target_layer_id, target_class, explainer_type)
        self.display_output(data_descr, target_layer_id, target_class, x, y, explainer_type, cams, output_prob, show)

    def get_item(self, data_descr):
        pt_id = data_descr["pt"]
        if pt_id in self.trainer.train_data.ids:
            dataset = self.trainer.train_data
        elif pt_id in self.trainer.test_data.ids:
            dataset = self.trainer.test_data
        else:
            dataset = self.trainer.val_data

        x, y = dataset.get_item(pt_id, data_descr["trial"])
        return x, y

    def display_output(self, data_descr, target_layer_id, target_class, x, y, explainer_type, maps, output_prob,
                       show=False):
        title = ("CAM for class " + str(target_class) + " (" + str(np.round(output_prob * 100, 2)) +
                 "%) - true label: " + str(int(y)))
        for block in x.keys():
            if not show:
                map = maps[block]
                plt.figure(figsize=(10, 10))
                if map.shape[1] > 1:
                    aspect = 0.05
                else:
                    aspect = 0.5
                plt.matshow(map, aspect=aspect, cmap=plt.get_cmap("jet"))
                plt.title(title)
                plt.xlabel(OpenFaceInstance.dim_names[block])
                if map.shape[1] > 1:
                    plt.xticks(range(map.shape[1]), OpenFaceInstance.dim_labels[block], rotation=45, fontsize=8)
                else:
                    plt.xticks([], [])
                item_name = data_descr["pt"] + "__trial" + str(data_descr["trial"])
                directory = self.jai_dir + self.model_name
                if item_name not in os.listdir(directory):
                    os.mkdir(directory + "/" + item_name)
                target_layer = "__conv_" + block + target_layer_id
                plt.savefig(directory + "/" + item_name + "/" + explainer_type.value + target_layer + "__" +
                            str(target_class) + ".png", format="png", bbox_inches="tight",
                            pad_inches=0, dpi=300)
                plt.close()
            else:
                # Project a proper function
                print("Functionality not available...")

    @staticmethod
    def draw_cam(trainer, x, target_layer_id, target_class, explainer_type):
        net = trainer.net
        net.set_training(False)
        net.set_cuda(False)
        x = {block: x[block].unsqueeze(0) for block in x.keys()}

        cams = {}
        for block in x.keys():
            target_layer = "conv_" + block + target_layer_id
            if isinstance(net.__dict__[target_layer], nn.Conv2d):
                is_2d = True
            elif isinstance(net.__dict__[target_layer], nn.Conv1d):
                is_2d = False
            else:
                is_2d = None
                print("The CAM method cannot be applied for the selected layer!")

            # Extract activations and gradients
            net.zero_grad()
            output, target_activation = net(x, layer_interrupt=target_layer)
            target_score = output[:, target_class]
            target_score.backward()
            target_grad = net.gradients

            # Compute CAM
            target_activation = target_activation.squeeze(0)
            target_grad = target_grad.squeeze(0)
            if explainer_type == ExplainerType.GC:
                cam = JAISystem.gc_map(target_activation, target_grad, is_2d)
            elif explainer_type == ExplainerType.HRC:
                cam = JAISystem.hrc_map(target_activation, target_grad)
            else:
                print("CAM generation method not implemented!")
                cam = None
            cam = cam.cpu().detach().numpy()
            cam = cam.transpose()
            cam = JAISystem.adjust_map(cam, x[block], is_2d)
            cams[block] = cam

        output_prob = torch.softmax(output, dim=1)
        output_prob = output_prob[:, target_class]
        output_prob = output_prob.detach().numpy()[0]
        return cams, output_prob

    @staticmethod
    def gc_map(target_activation, target_grad, is_2d=True):
        if is_2d:
            dim_mean = (1, 2)
        else:
            dim_mean = 1
        weights = torch.mean(target_grad, dim=dim_mean)

        for i in range(target_activation.shape[0]):
            target_activation[i] *= weights[i]
        cam = torch.sum(target_activation, dim=0)

        cam = torch.relu(cam)
        return cam

    @staticmethod
    def hrc_map(target_activation, target_grad):
        for i in range(target_activation.shape[0]):
            target_activation[i] *= target_grad[i]
        cam = torch.sum(target_activation, dim=0)

        cam = torch.relu(cam)
        return cam

    @staticmethod
    def adjust_map(map, x, is_2d=True):
        map = JAISystem.normalize_map(map)

        if is_2d:
            dim_reshape = (x.shape[2], x.shape[1])
        else:
            dim_reshape = (1, x.shape[1])
        map = cv2.resize(map, dim_reshape)
        return map

    @staticmethod
    def normalize_map(map):
        maximum = np.max(map)
        minimum = np.min(map)
        if maximum == minimum:
            if maximum == 1:
                map = np.ones(map.shape)
            else:
                map = np.zeros(map.shape)
        else:
            map = (map - minimum) / (maximum + minimum)

        map = np.uint8(255 * map)
        return map


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../../"

    # Define the system
    model_name1 = "stimulus_conv2d"
    system1 = JAISystem(working_dir=working_dir1, model_name=model_name1)

    # Explain one item
    data_descr1 = {"pt": "bam2_001", "trial": 3}
    target_layer_id1 = "0"
    target_class1 = 1
    explainer_type1 = ExplainerType.GC
    show1 = False
    system1.get_cam(data_descr=data_descr1, target_layer_id=target_layer_id1, target_class=target_class1,
                    explainer_type=explainer_type1, show=show1)
