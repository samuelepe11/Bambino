# Import packages
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn

from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.ToyOpenFaceDataset import ToyOpenFaceDataset
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset
from DataUtils.OpenFaceInstance import OpenFaceInstance
from TrainUtils.NetworkTrainer import NetworkTrainer
from Types.ExplainerType import ExplainerType
from Types.SetType import SetType


# Class
class JAISystem:
    # Define class attributes
    time_steps = list(range(0, OpenFaceDataset.max_time * OpenFaceDataset.fc + 50, 50))

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
        self.results_dir = working_dir + results_fold
        self.models_dir = self.results_dir + OpenFaceDataset.models_fold
        self.jai_dir = self.results_dir + OpenFaceDataset.jai_fold
        self.is_toy = is_toy
        self.is_boa = is_boa

        self.model_name = model_name
        if model_name not in os.listdir(self.jai_dir):
            os.mkdir(self.jai_dir + model_name)

        self.trainer = NetworkTrainer.load_model(working_dir=working_dir, model_name=model_name, trial_n=trial_n,
                                                 use_cuda=use_cuda, is_toy=is_toy, is_boa=is_boa)
        self.trainer.show_model()

    def get_cam(self, data_descr, target_layer_id, target_class, explainer_type, show=False, show_graphs=False,
                x=None, y=None, cams_in_one_plot=False, normalize_jointly=False):
        if data_descr is not None:
            x, y = self.get_item(data_descr)
        cams, output_prob = self.draw_cam(self.trainer, x, target_layer_id, target_class, explainer_type)
        real_signal_length = x["g"].shape[0] if self.is_toy and not self.is_boa else None
        if self.is_toy or self.is_boa:
            initial_pad_dim = OpenFaceDataset.time_stimulus * OpenFaceDataset.fc
            x = {k: np.concatenate([np.zeros((initial_pad_dim, x[k].shape[1]), np.uint8), v]) for k, v in x.items()}
            cams = {k: np.concatenate([np.zeros((initial_pad_dim, cams[k].shape[1]), np.uint8), v])
                    for k, v in cams.items()}
            if not self.is_boa:
                final_pad_dim = OpenFaceDataset.max_time * OpenFaceDataset.fc - cams["g"].shape[0]
                x = {k: np.concatenate([v, np.zeros((final_pad_dim, x[k].shape[1]), np.uint8)]) for k, v in x.items()}
                cams = {k: np.concatenate([v, np.zeros((final_pad_dim, cams[k].shape[1]), np.uint8)]) for k, v in cams.items()}

        if data_descr is not None:
            self.display_output(data_descr, target_layer_id, target_class, x, y, explainer_type, cams, output_prob,
                                show, show_graphs, cams_in_one_plot=cams_in_one_plot,
                                normalize_jointly=normalize_jointly, real_signal_length=real_signal_length)
        else:
            return cams

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
                       normalize_jointly=False, real_signal_length=None):
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
            maps = JAISystem.normalize_jointly(maps)

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

                    # Highlight ending time
                    if real_signal_length is not None:
                        plt.axvline(x=OpenFaceDataset.time_stimulus * OpenFaceDataset.fc + real_signal_length,
                                    color="black", linestyle="--", linewidth=2)
                        plt.text(x=OpenFaceDataset.time_stimulus * OpenFaceDataset.fc + real_signal_length,
                                 y=map.shape[1], s="acquisition end", color="black", ha="center", va="top")

                    # Adjust axes
                    plt.title(title)
                    plt.xlabel("Time (s)")
                    plt.ylabel(OpenFaceInstance.dim_names[block])
                    plt.xticks(JAISystem.time_steps, [str(int(t / OpenFaceDataset.fc)) for t in JAISystem.time_steps],
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

                    # Highlight ending time
                    if real_signal_length is not None:
                        axs[count].axvline(x=OpenFaceDataset.time_stimulus * OpenFaceDataset.fc + real_signal_length,
                                           color="black", linestyle="--", linewidth=2)
                        axs[count].text(x=OpenFaceDataset.time_stimulus * OpenFaceDataset.fc + real_signal_length,
                                        y=map.shape[1], s="acquisition end", color="black", ha="center", va="top")

                    # Adjust axes
                    axs[count].set_xlabel("Time (s)", loc="right")
                    axs[count].set_title(OpenFaceInstance.dim_names[block])
                    axs[count].set_xticks(JAISystem.time_steps,
                                          [str(int(t / OpenFaceDataset.fc)) for t in JAISystem.time_steps],
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
                    if self.is_toy or self.is_boa:
                        x[block] = x[block][OpenFaceDataset.time_stimulus * OpenFaceDataset.fc:]
                        map = map[OpenFaceDataset.time_stimulus * OpenFaceDataset.fc:]
                    if real_signal_length is not None:
                        x[block] = x[block][:real_signal_length]
                        map = map[:real_signal_length]
                    JAISystem.show_graphs(item=x, block=block, map=map, name_start=name_start,
                                          real_signal_length=real_signal_length)
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
    def show_graphs(item, block, map, name_start, real_signal_length=None):
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
        time_steps = np.arange(OpenFaceDataset.max_time * OpenFaceDataset.fc) if real_signal_length is None \
            else np.arange(real_signal_length)
        time_ticks = JAISystem.time_steps if real_signal_length is None else list(range(0, real_signal_length + 10, 10))
        time_ticks_labels = [str(int(t / OpenFaceDataset.fc)) for t in time_ticks] if real_signal_length is None \
            else [str(t / OpenFaceDataset.fc) for t in time_ticks]
        s = 40 if real_signal_length is None else 60
        for i in range(x.shape[1]):
            plt.subplot(settings[2], settings[3], i + 1)
            plt.tight_layout()
            plt.plot(time_steps, x[:, i], color="black", linewidth=0.5)
            norm = mcolors.Normalize(vmin=0, vmax=0.1) if len(np.unique(map[:, i])) == 1 and map[0][i] == 0 else None
            plt.scatter(time_steps, x[:, i], c=map[:, i], cmap="jet", marker=".", s=s, norm=norm)
            plt.colorbar()
            plt.xticks(time_ticks, time_ticks_labels, fontsize=8)
            plt.xlabel("Time (s)")
            plt.title(labels[i].upper())

            # Highlight stimulus time
            if real_signal_length is None:
                plt.axvline(x=OpenFaceDataset.time_stimulus * OpenFaceDataset.fc, color="black", linestyle="--",
                            linewidth=2)

        plt.savefig(name_start + ".png", format="png", bbox_inches="tight", pad_inches=0, dpi=500)
        plt.close()

    @staticmethod
    def draw_cam(trainer, x, target_layer_id, target_class, explainer_type):
        net = trainer.net
        net.set_training(False)
        net.set_cuda(False)
        x = {key: x[key].unsqueeze(0) for key in x.keys()}
        x = {key: (x[key] - trainer.train_mean[key]) / trainer.train_std[key] for key in x.keys()}

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
            map = (map - minimum) / (maximum - minimum)

        map = np.uint8(255 * map)
        return map

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
    model_name1 = "stimulus_conv2d_optuna"
    trial_n1 = 44
    use_cuda1 = False
    is_toy1 = True
    is_boa1 = False
    system1 = JAISystem(working_dir=working_dir1, model_name=model_name1, trial_n=trial_n1, is_toy=is_toy1,
                        is_boa=is_boa1)

    # Explain some items
    data_descriptions = [{"pt": "bam2_004", "trial": 7}, {"pt": "bam2_004", "trial": 9},
                         {"pt": "bam2_010", "trial": 16}, {"pt": "bam2_020", "trial": 32}]
    target_layer_id1 = "1"
    target_classes = [0, 1]
    explainer_types = [ExplainerType.GC, ExplainerType.HRC]
    show1 = False
    show_graphs1 = True
    cams_in_one_plot1 = False
    normalize_jointly1 = False
    for data_descr1 in data_descriptions:
        for target_class1 in target_classes:
            for explainer_type1 in explainer_types:
                system1.get_cam(data_descr=data_descr1, target_layer_id=target_layer_id1, target_class=target_class1,
                                explainer_type=explainer_type1, show=show1, show_graphs=show_graphs1, 
                                cams_in_one_plot=cams_in_one_plot1, normalize_jointly=normalize_jointly1)

    # Average children explanations
    '''set_types = [SetType.VAL, SetType.TEST]
    explainer_type1 = ExplainerType.GC
    for set_type1 in set_types:
        system1.average_explanations(set_type=set_type1, explainer_type=explainer_type1,
                                     target_layer_id=target_layer_id1, cams_in_one_plot=cams_in_one_plot1,
                                     normalize_jointly=normalize_jointly1)'''
