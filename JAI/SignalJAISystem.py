# Import packages
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
from signal_grad_cam import TorchCamBuilder

from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.ToyOpenFaceDataset import ToyOpenFaceDataset
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset
from DataUtils.OpenFaceInstance import OpenFaceInstance
from TrainUtils.NetworkTrainer import NetworkTrainer
from Types.ExplainerType import ExplainerType
from Types.SetType import SetType


# Class
class SignalJAISystem:

    def __init__(self, working_dir, model_name, trial_n=None, use_cuda=False, is_toy=False, is_boa=False):
        # Initialize attributes
        self.working_dir = working_dir
        if is_toy:
            results_fold = ToyOpenFaceDataset.results_fold
        elif is_boa:
            results_fold = BoaOpenFaceDataset.results_fold
        else:
            results_fold = OpenFaceDataset.results_fold
        self.results_dir = working_dir + results_fold
        self.models_dir = self.results_dir + OpenFaceDataset.models_fold
        self.jai_dir = self.results_dir + OpenFaceDataset.jai_fold
        self.is_toy = is_toy
        self.is_boa = is_boa

        self.model_name = model_name
        if model_name not in os.listdir(self.jai_dir):
            os.mkdir(self.jai_dir + model_name)

        self.trainer = NetworkTrainer.load_model(working_dir=working_dir, model_name=model_name, trial_n=trial_n,
                                                 use_cuda=use_cuda, is_toy=True)
        self.cam_builder = TorchCamBuilder(model=self.trainer.net, transform_fn=None, time_axs=0,
                                           class_names=OpenFaceDataset.trial_types, use_gpu=use_cuda, extend_search=True)

    def get_cam(self, data_description, explainer_types, target_layers_ids, softmax_final):
        x, y = self.get_item(data_description)
        data_list = [np.concatenate([xi for xi in x.values()])]
        data_labels = [y]
        data_names = [data_description.keys()[0] + "_" + str(data_description.values()[0])]
        cams, predicted_probs, bar_ranges = self.cam_builder.get_cam(data_list, data_labels=data_labels,
                                                                     target_classes=target_classes,
                                                                     explainer_types=explainer_types,
                                                                     target_layers=target_layers_names,
                                                                     softmax_final=softmax_final, data_names=data_names,
                                                                     results_dir_path=results_dir,
                                                                     data_sampling_freq=OpenFaceDataset.fc)
        return cams, predicted_probs, bar_ranges

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
                       show=False, show_graphs=False, averaged_folder=None, cams_in_one_plot=False,
                       normalize_jointly=False):
        if averaged_folder is None:
            title = ("CAM for class " + str(target_class) + " (" + str(np.round(output_prob * 100, 2)) +
                     "%) - true label: " + str(int(y)))
        elif data_descr is not None:
            title = "Averaged CAM for patient " + data_descr + " (class " + str(target_class) + ")"
        else:
            title = "Averaged CAM for class " + str(target_class)

        if cams_in_one_plot:
            if maps["h"].shape[1] > 1:
                #fig_size = (4, 15)
                fig_size = (21, 5)
            else:
                fig_size = (10, 5)
            plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4)
            #fig, axs = plt.subplots(nrows=3, ncols=1, figsize=fig_size)
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=fig_size)
            count = 0

        if normalize_jointly:
            maps = SignalJAISystem.normalize_jointly(maps)

        v_max = 255 if not normalize_jointly else 1
        fig_size = (50, 50) if maps["h"].shape[1] > 1 else (50, 2)
        aspect = 5 if maps["h"].shape[1] > 1 else 10
        for block in maps.keys():
            if not show:
                map = maps[block]
                if not cams_in_one_plot:
                    plt.figure(figsize=fig_size)

                if (len(np.unique(map)) == 1 and np.unique(map) == 0) or normalize_jointly:
                    norm = mcolors.Normalize(vmin=0, vmax=v_max)
                else:
                    norm = None
                if not cams_in_one_plot:
                    plt.matshow(np.transpose(map), aspect=aspect, cmap=plt.get_cmap("jet"), norm=norm)

                    # Add colorbar
                    plt.colorbar()

                    # Highlight stimulus time
                    plt.axvline(x=OpenFaceDataset.time_stimulus * OpenFaceDataset.fc, color="black", linestyle="--",
                                linewidth=2)
                    plt.text(x=OpenFaceDataset.time_stimulus * OpenFaceDataset.fc, y=map.shape[1],
                             s="possible stimulus", color="black", ha="center", va="top")

                    # Adjust axes
                    plt.title(title)
                    plt.xlabel("Time (s)")
                    plt.ylabel(OpenFaceInstance.dim_names[block])
                    plt.xticks(SignalJAISystem.time_steps, [str(int(t / OpenFaceDataset.fc)) for t in SignalJAISystem.time_steps],
                               fontsize=8)
                    if map.shape[1] > 1:
                        plt.yticks(range(map.shape[1]), [s.upper() for s in OpenFaceInstance.dim_labels[block]],
                                   rotation=0, fontsize=7)
                    else:
                        plt.yticks([], [])
                else:
                    aspect = 20 if map.shape[1] > 1 else 10
                    axs[count].matshow(np.transpose(map), aspect=aspect, cmap=plt.get_cmap("jet"), norm=norm)

                    # Highlight stimulus time
                    axs[count].axvline(x=OpenFaceDataset.time_stimulus * OpenFaceDataset.fc, color="black", linestyle="--",
                                       linewidth=2)
                    axs[count].text(x=OpenFaceDataset.time_stimulus * OpenFaceDataset.fc, y=map.shape[1],
                                    s="possible stimulus", color="black", ha="center", va="top")

                    # Adjust axes
                    axs[count].set_xlabel("Time (s)", loc="right")
                    axs[count].set_title(OpenFaceInstance.dim_names[block])
                    axs[count].set_xticks(SignalJAISystem.time_steps,
                                          [str(int(t / OpenFaceDataset.fc)) for t in SignalJAISystem.time_steps],
                                          fontsize=8)

                    if map.shape[1] > 1:
                        axs[count].set_yticks(range(map.shape[1]),
                                              [s.upper() for s in OpenFaceInstance.dim_labels[block]], rotation=0,
                                              fontsize=7)
                        '''if count == 0:
                            # Adjust vertical distancing of the plots
                            box = axs[0].get_position()
                            new_box = [box.x0, box.y0 - 0.06, box.width, box.height]
                            axs[0].set_position(new_box)'''

                    else:
                        axs[count].set_yticks([], [])

                    count += 1
                if averaged_folder is None:
                    item_name = data_descr["pt"] + "__trial" + str(data_descr["trial"])
                else:
                    item_name = averaged_folder
                directory = self.jai_dir + self.model_name
                if averaged_folder is None and item_name not in os.listdir(directory):
                    os.mkdir(directory + "/" + item_name)
                target_layer = "__conv_" + block + target_layer_id

                if not cams_in_one_plot or count == 3:
                    addon = target_layer if not cams_in_one_plot else ""
                    path = directory + "/" + item_name + "/" + explainer_type.value + addon + "__" + str(target_class)
                    plt.savefig(path + ".png", format="png", bbox_inches="tight", pad_inches=0, dpi=300)
                    plt.close()

                if show_graphs and averaged_folder is None and not cams_in_one_plot:
                    name_start = directory + "/" + item_name + "/" + explainer_type.value + "/"
                    if explainer_type.value not in os.listdir(directory + "/" + item_name):
                        os.mkdir(name_start)
                    name_start += "conv_" + block + target_layer_id + "__" + str(target_class)
                    SignalJAISystem.show_graphs(item=x, block=block, map=map, name_start=name_start)
            else:
                # Project a proper function
                print("Functionality not available...")

    def average_explanations(self, set_type, explainer_type, target_layer_id, cams_in_one_plot=False,
                             normalize_jointly=False):
        if set_type == SetType.TRAIN:
            dataset = self.trainer.train_data
        elif set_type == SetType.TEST:
            dataset = self.trainer.test_data
        else:
            dataset = self.trainer.val_data

        # Create folders for storage
        children_folder = set_type.value + "_children_averaged"
        if children_folder not in os.listdir(self.jai_dir + self.model_name):
            os.mkdir(self.jai_dir + self.model_name + "/" + children_folder)
        class_folder = set_type.value + "_class_averaged"
        if class_folder not in os.listdir(self.jai_dir + self.model_name):
            os.mkdir(self.jai_dir + self.model_name + "/" + class_folder)

        class_cams_list = [[], []]
        for pt_id in dataset.ids:
            if pt_id not in os.listdir(self.jai_dir + self.model_name + "/" + children_folder):
                os.mkdir(self.jai_dir + self.model_name + "/" + children_folder + "/" + pt_id)

            for label in [0, 1]:
                cams_list = []
                for i, instance in enumerate(dataset.instances):
                    if instance.pt_id == pt_id:
                        x, y, _ = dataset.__getitem__(i)
                        cams = self.get_cam(data_descr=None, target_layer_id=target_layer_id, target_class=label,
                                            explainer_type=explainer_type, show=False, show_graphs=False, x=x, y=y)
                        if y == label:
                            cams_list.append(cams)
                            class_cams_list[label].append(cams)

                # Get child averaged maps
                if len(cams_list) > 0:
                    cams = {"g": np.mean([cam["g"] for cam in cams_list], axis=0),
                            "h": np.mean([cam["h"] for cam in cams_list], axis=0),
                            "f": np.mean([cam["f"] for cam in cams_list], axis=0)}
                    self.display_output(data_descr=pt_id, target_layer_id=target_layer_id, target_class=label, x=None,
                                        y=None, explainer_type=explainer_type, maps=cams, output_prob=None,
                                        show=False, show_graphs=False, averaged_folder=children_folder + "/" + pt_id,
                                        cams_in_one_plot=cams_in_one_plot, normalize_jointly=normalize_jointly)

            # Get class averaged maps
            for label in [0, 1]:
                if len(class_cams_list[label]) > 0:
                    cams = {"g": np.mean([cam["g"] for cam in class_cams_list[label]], axis=0),
                            "h": np.mean([cam["h"] for cam in class_cams_list[label]], axis=0),
                            "f": np.mean([cam["f"] for cam in class_cams_list[label]], axis=0)}
                    self.display_output(data_descr=None, target_layer_id=target_layer_id, target_class=label, x=None,
                                        y=None, explainer_type=explainer_type, maps=cams, output_prob=None, show=False,
                                        show_graphs=False, averaged_folder=class_folder,
                                        cams_in_one_plot=cams_in_one_plot, normalize_jointly=normalize_jointly)

    @staticmethod
    def show_graphs(item, block, map, name_start):
        # Extract signals
        x = item[block]
        name = OpenFaceInstance.dim_names[block]
        labels = OpenFaceInstance.dim_labels[block]

        if map.shape[1] == 1:
            map = np.tile(map, (1, x.shape[1]))

        # Draw single plots
        settings = OpenFaceInstance.subplot_settings[block]
        plt.figure(figsize=(settings[0], settings[1]))
        plt.suptitle(name.upper())
        time_steps = np.arange(OpenFaceDataset.max_time * OpenFaceDataset.fc)
        for i in range(x.shape[1]):
            plt.subplot(settings[2], settings[3], i + 1)
            plt.tight_layout()
            plt.plot(time_steps, x[:, i], color="black", linewidth=0.5)
            norm = mcolors.Normalize(vmin=0, vmax=0.1) if len(np.unique(map[:, i])) == 1 and map[0][i] == 0 else None
            plt.scatter(time_steps, x[:, i], c=map[:, i], cmap="jet", marker=".", s=40, norm=norm)
            plt.colorbar()
            plt.xticks(SignalJAISystem.time_steps, [str(int(t / OpenFaceDataset.fc)) for t in SignalJAISystem.time_steps],
                       fontsize=8)
            plt.xlabel("Time (s)")
            plt.title(labels[i].upper())

            # Highlight stimulus time
            plt.axvline(x=OpenFaceDataset.time_stimulus * OpenFaceDataset.fc, color="black", linestyle="--",
                        linewidth=2)

        plt.savefig(name_start + ".png", format="png", bbox_inches="tight", pad_inches=0, dpi=500)
        plt.close()

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
    def normalize_jointly(maps):
        maximum = np.max([np.max(map) for map in maps.values()])
        maps = {k: v / maximum for k, v in maps.items()}

        return maps


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../../"

    # Define the system
    model_name1 = "stimulus_conv1d_optuna"
    trial_n1 = 39
    use_cuda1 = False
    is_toy1 = True
    is_boa1 = False
    system1 = SignalJAISystem(working_dir=working_dir1, model_name=model_name1, trial_n=trial_n1, is_toy=is_toy1,
                              is_boa=is_boa1)

    # Explain some items
    data_descriptions = [{"pt": "bam2_004", "trial": 7}, {"pt": "bam2_004", "trial": 9},
                         {"pt": "bam2_010", "trial": 16}, {"pt": "bam2_020", "trial": 32}]
    target_layer_id1 = "2"
    target_classes = [0, 1]
    explainer_types1 = [ExplainerType.GC, ExplainerType.HRC]
    softmax_final1 = True
    '''show1 = False
    show_graphs1 = True
    cams_in_one_plot1 = True
    normalize_jointly1 = True'''
    for data_descr1 in data_descriptions:
        cams, predicted_probs, bar_ranges = system1.get_cam(data_description=data_descr1,
                                                            explainer_types=explainer_types1,
                                                            target_layers_ids=target_layer_id1,
                                                            softmax_final=softmax_final1)

    # Average children explanations
    '''set_types = [SetType.VAL, SetType.TEST]
    explainer_type1 = ExplainerType.GC
    for set_type1 in set_types:
        system1.average_explanations(set_type=set_type1, explainer_type=explainer_type1,
                                     target_layer_id=target_layer_id1, cams_in_one_plot=cams_in_one_plot1,
                                     normalize_jointly=normalize_jointly1)'''
