# Import packages
import os

import pandas as pd
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from collections import Counter

from DataUtils.OpenFaceInstance import OpenFaceInstance
from Types.TaskType import TaskType


# Class
class OpenFaceDataset(Dataset):
    # Define class attributes
    trial_types = ["control", "stimulus"]
    age_groups = ["7-11", "12-18", "19-24"]
    trial_id_groups = ["0-30th percentiles", "30-70th percentiles", "70-100th percentiles"]

    time_stimulus = 2
    max_time = 12
    fc = 25

    # Define folders
    data_fold = "data/"
    preliminary_fold = "preliminary_analysis/"
    results_fold = "results/"
    models_fold = "models/"
    jai_fold = "JAI/"

    def __init__(self, dataset_name, working_dir, file_name, data_instances=None):
        self.working_dir = working_dir
        self.data_dir = working_dir + self.data_fold
        self.results_dir = working_dir + self.results_fold
        self.file_name = file_name
        self.dataset_name = dataset_name
        self.preliminary_dir = self.results_dir + self.preliminary_fold
        if self.dataset_name not in os.listdir(self.preliminary_dir):
            os.mkdir(self.preliminary_dir + self.dataset_name)
        self.preliminary_dir += self.dataset_name + "/"

        # Read file
        data = pd.read_csv(self.data_dir + file_name, delimiter=",")

        # Adjust values
        data.loc[data["sex"] == "Boy", "sex"] = 1
        data.loc[data["sex"] == "Girl", "sex"] = 0
        data.loc[data["trial_type"] == self.trial_types[1], "trial_type"] = 1
        data.loc[data["trial_type"] == self.trial_types[0], "trial_type"] = 0
        data.loc[(data["trial_outcome"] == "true_positive") |
                 (data["trial_outcome"] == "false_positive"), "trial_outcome"] = 1
        data.loc[(data["trial_outcome"] == "true_negative") |
                 (data["trial_outcome"] == "false_negative"), "trial_outcome"] = 0

        # Read separate trials
        if data_instances is None:
            self.ids = np.unique(data["participant_id"])
            self.trials = np.unique(data["trial_id"])
            self.instances = []
            for pt_id in self.ids:
                for trial in self.trials:
                    temp_data = data.loc[(data["participant_id"] == pt_id) & (data["trial_id"] == trial), :]
                    if temp_data.shape[0] == 0:
                        continue
                    self.instances.append(OpenFaceInstance(temp_data))
        else:
            self.ids = np.unique([instance.pt_id for instance in data_instances])
            self.instances = data_instances
        self.len = len(self.instances)

        self.task_type = None
        self.clinician_cm = None
        self.clinician_stats = None
        self.trial_id_stats = None

    def __getitem__(self, idx):
        instance = self.instances[idx]

        x = {"g": torch.Tensor(instance.gaze_info),
             "h": torch.Tensor(instance.head_info),
             "f": torch.Tensor(instance.face_info)}
        y = OpenFaceDataset.preprocess_label(instance.trial_type)

        age = OpenFaceInstance.categorize_age(instance.age)
        age = OpenFaceDataset.preprocess_label(age)
        trial_id_categorical = OpenFaceInstance.categorize_trial_id(instance.trial_id, self.trial_id_stats)
        trial_id_categorical = OpenFaceDataset.preprocess_label(trial_id_categorical)
        trial_id = OpenFaceDataset.preprocess_label(instance.trial_id)
        return x, y, [age, trial_id_categorical, trial_id]

    def __len__(self):
        return self.len

    def get_extra_info(self, x):
        ages = []
        trial_ids = []
        for i in range(x["g"].shape[0]):
            gi = x["g"][i, :, :]
            hi = x["h"][i, :, :]
            fi = x["f"][i, :, :]
            for instance in self.instances:
                if instance.gaze_info == gi and instance.head_info == hi and instance.face_info == fi:
                    ages.append(OpenFaceInstance.categorize_age(instance.age))
                    trial_ids.append(OpenFaceInstance.categorize_trial_id(instance.trial_id, self.trial_id_stats))

        return torch.Tensor(ages), torch.Tensor(trial_ids)

    def split_dataset(self, train_perc):
        # Get training set
        n_pt = len(self.ids)
        n_train = int(np.round(n_pt * train_perc))
        id_train = random.sample(list(self.ids), n_train)
        train_instances = self.get_instances(id_train)
        train_set = OpenFaceDataset(dataset_name="training_set", working_dir=self.working_dir, file_name=self.file_name,
                                    data_instances=train_instances)
        train_set.store_dataset()

        # Get validation set
        n_val = int(np.floor((n_pt - n_train) / 2))
        remaining = list(set(self.ids) - set(id_train))
        id_val = random.sample(remaining, n_val)
        val_instances = self.get_instances(id_val)
        val_set = OpenFaceDataset(dataset_name="validation_set", working_dir=self.working_dir, file_name=self.file_name,
                                  data_instances=val_instances)
        val_set.store_dataset()

        # Get test set
        id_test = list(set(remaining) - set(id_val))
        test_instances = self.get_instances(id_test)
        test_set = OpenFaceDataset(dataset_name="test_set", working_dir=self.working_dir, file_name=self.file_name,
                                   data_instances=test_instances)
        test_set.store_dataset()

    def get_instances(self, ids):
        return [instance for instance in self.instances if instance.pt_id in ids]

    def compute_statistics(self, trial_id_stats=None):
        # Count data
        print("Number of patients:", len(self.ids))
        print("Number of data instances:",  self.len)

        # Count trial types
        n_stimulus = len([instance for instance in self.instances if instance.trial_type == 1])
        n_control = self.len - n_stimulus
        OpenFaceDataset.draw_pie_bar_plot([n_control, n_stimulus], self.trial_types, "Trial types",
                                          self.preliminary_dir + "trial_types")
        OpenFaceDataset.draw_pie_bar_plot([n_control, n_stimulus], self.trial_types, "Trial types",
                                          self.preliminary_dir + "trial_types", do_bar=True)

        # Count age (at patient level)
        ages = []
        ages_categorical = []
        for pt_id in self.ids:
            for instance in self.instances:
                if instance.pt_id == pt_id:
                    age = instance.age
                    ages.append(age)
                    ages_categorical.append(OpenFaceInstance.categorize_age(age))
                    break
        OpenFaceDataset.draw_hist(ages, 24, "Age distribution (in months)", self.preliminary_dir +
                                  "age_distr")
        OpenFaceDataset.draw_hist(ages_categorical, 3, "Age distribution (categorical)",
                                  self.preliminary_dir + "age_categ_distr", self.age_groups)

        # Count number of trials
        trials = [instance.trial_id for instance in self.instances]
        OpenFaceDataset.draw_hist(trials, 90, "Trial distribution", self.preliminary_dir + "trial_distr")

        # Count categorized number of trials
        if trial_id_stats is None:
            # Compute trial_id mean and std
            m_trial = np.mean(trials)
            s_trial = np.std(trials)
            trial_id_stats = (m_trial, s_trial)
            self.trial_id_stats = trial_id_stats
        trials_categorical = [OpenFaceInstance.categorize_trial_id(trial, trial_id_stats) for trial in trials]
        OpenFaceDataset.draw_hist(trials_categorical, 3, "Trial distribution (categorical)",
                                  self.preliminary_dir + "trial_categ_distr", self.trial_id_groups)

        # Count instances by both age and number of trials
        ages_categorical_all = [OpenFaceInstance.categorize_age(instance.age) for instance in self.instances]
        pairs = list(zip(ages_categorical_all, trials_categorical))
        pair_counts = Counter(pairs)
        cm = np.zeros((3, 3), dtype=int)
        for k, v in pair_counts.items():
            cm[k[0], k[1]] = int(v)
        plt.figure(figsize=(8, 4))
        plt.imshow(cm, cmap="Reds")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                plt.text(j, i, f"{val.item()}", ha="center", va="center", color="black", fontsize="xx-large")
        plt.title("Age (categorical) vs. Trial ID (categorical)")
        plt.xticks(list(range(3)), OpenFaceDataset.trial_id_groups, rotation=10)
        plt.xlabel("Trial ID")
        plt.yticks(list(range(3)), OpenFaceDataset.age_groups)
        plt.ylabel("Age")
        plt.savefig(self.preliminary_dir + "age_vs_trial.jpg", dpi=300, bbox_inches="tight")
        plt.close()

        # Count sequence duration
        durations = [instance.face_info.shape[0] / OpenFaceDataset.fc for instance in self.instances]
        print("Mean sequence duration = ", np.mean(durations), "s", "(std = " + str(np.std(durations)) + ")")
        for instance in self.instances:
            length = instance.face_info.shape[0]
            if length < 300:
                print("Patient", instance.pt_id, "(trial " + str(instance.trial_id) + ") has", length, "data points")

    def remove_short_sequences(self):
        removable = []
        for instance in self.instances:
            if instance.face_info.shape[0] < 300:
                removable.append(instance)

        for rm in removable:
            self.instances.remove(rm)
        self.len = len(self.instances)

    def store_dataset(self):
        file_path = self.data_dir + self.dataset_name + ".pt"
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
        print("The dataset '" + self.dataset_name + "' have been stored!")

    def get_item(self, pt_id, trial_id):
        idx = None
        for i in range(len(self.instances)):
            instance = self.instances[i]
            if instance.pt_id == pt_id and instance.trial_id == trial_id:
                idx = i
        if idx is None:
            print("Trial " + trial_id + " of patient " + pt_id + " not found!")
            return None
        else:
            x, y, _ = self.__getitem__(idx)
            return x, y

    @staticmethod
    def preprocess_label(y):
        y = torch.Tensor([y])
        return y.to(torch.long)

    @staticmethod
    def draw_pie_bar_plot(data, labels, title, file_name, do_bar=False):
        if file_name is not None:
            plt.figure()

        if not do_bar:
            plt.pie(data, labels=labels, autopct="%1.1f%%")
        else:
            plt.bar(labels, data)
            plt.ylabel("Absolute frequency")
        plt.title(title)

        if file_name is not None:
            plt.savefig(file_name + ".jpg", dpi=300)
            plt.close()

    @staticmethod
    def draw_hist(data, maximum, title, file_name, labels=None):
        if file_name is not None:
            plt.figure()

        if maximum > 3:
            bins = int(maximum / 3)
        else:
            bins = maximum
        ax = plt.hist(data, bins=bins)

        if labels is None:
            plt.xticks(range(0, maximum + 1, 4))
        else:
            plt.xticks([0.3, 1, 1.7], ["", "", ""])
        plt.ylabel("Absolute frequency")
        plt.title(title)

        if labels is not None:
            rects = ax[2].patches
            for rect, label in zip(rects, labels):
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
                         ha="center", va="bottom")

        if file_name is not None:
            plt.savefig(file_name + ".jpg", dpi=300)
            plt.close()

    @staticmethod
    def load_dataset(working_dir, dataset_name, task_type=None, train_trial_id_stats=None):
        file_path = working_dir + OpenFaceDataset.data_fold + dataset_name + ".pt"
        with open(file_path, "rb") as file:
            dataset = pickle.load(file)
        dataset.task_type = task_type

        if train_trial_id_stats is not None:
            dataset.trial_id_stats = train_trial_id_stats

        return dataset


# Main
if __name__ == "__main__":
    # Define seeds
    seed = 1
    random.seed(seed)

    # Define variables
    working_dir1 = "./../../"
    file_name1 = "processed_openface_toy_data.csv"
    dataset_name1 = "complete_dataset"

    # Read data
    # dataset1 = OpenFaceDataset(dataset_name=dataset_name1, working_dir=working_dir1, file_name=file_name1)

    # Compute statistics
    # dataset1.compute_statistics()

    # Remove short sequences
    # dataset1.remove_short_sequences()

    # Divide dataset
    train_perc1 = 0.7
    # dataset1.split_dataset(train_perc=train_perc1)

    # Load training set
    print()
    train_set1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="training_set")
    train_set1.compute_statistics()
    # train_set1.store_dataset()

    # Load training set
    print()
    val_set1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="validation_set")
    val_set1.compute_statistics(train_set1.trial_id_stats)

    # Load training set
    print()
    test_set1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="test_set")
    test_set1.compute_statistics(train_set1.trial_id_stats)

