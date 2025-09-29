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
    data_fold = "data/preliminary_toy/"
    preliminary_fold = "preliminary_analysis/"
    results_fold = "results/preliminary_toy/"
    models_fold = "models/"
    jai_fold = "JAI/"

    def __init__(self, dataset_name, working_dir, file_name, data_instances=None, is_boa=False, is_toy=False):
        self.working_dir = working_dir
        self.data_dir = working_dir + self.data_fold
        self.results_dir = working_dir + self.results_fold
        self.file_name = file_name
        self.dataset_name = dataset_name
        self.is_boa = is_boa
        self.is_toy = is_toy
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
        if not is_boa:
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
                    if (temp_data.shape[0] == 0 or temp_data["low_confidence_for_trial"].iloc[0] or
                            (temp_data["confidence"].iloc[0:52] <= 0.5).all()):
                        continue
                    self.instances.append(OpenFaceInstance(temp_data, is_boa, is_toy))
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

        try:
            age_categorical = OpenFaceInstance.categorize_age(instance.age, self.is_boa)
        except AttributeError:
            age_categorical = OpenFaceInstance.categorize_age(instance.age)
        age_categorical = OpenFaceDataset.preprocess_label(age_categorical)

        trial_id_stats = self.trial_id_stats if not isinstance(self.trial_id_stats, dict) else self.trial_id_stats[instance.pt_id]
        trial_id_categorical = OpenFaceInstance.categorize_trial_id(instance.trial_id, trial_id_stats)
        trial_id_categorical = OpenFaceDataset.preprocess_label(trial_id_categorical)
        trial_id = OpenFaceDataset.preprocess_label(instance.trial_id)
        return x, y, [age_categorical, trial_id_categorical, trial_id]

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
                    ages.append(OpenFaceInstance.categorize_age(instance.age, self.is_boa))
                    trial_ids.append(OpenFaceInstance.categorize_trial_id(instance.trial_id, self.trial_id_stats))

        return torch.Tensor(ages), torch.Tensor(trial_ids)

    def split_dataset(self, train_perc, is_child_dataset=False):
        # Get training set instances
        n_pt = len(self.ids)
        n_train = int(np.round(n_pt * train_perc))
        id_train = random.sample(list(self.ids), n_train)
        train_instances = self.get_instances(id_train)

        # Get validation set instances
        n_val = int(np.floor((n_pt - n_train) / 2))
        remaining = list(set(self.ids) - set(id_train))
        id_val = random.sample(remaining, n_val)
        val_instances = self.get_instances(id_val)

        # Get test set instances
        id_test = list(set(remaining) - set(id_val))
        test_instances = self.get_instances(id_test)

        if not is_child_dataset:
            # Create and store datasets
            train_set = OpenFaceDataset(dataset_name="training_set", working_dir=self.working_dir,
                                        file_name=self.file_name,
                                        data_instances=train_instances)
            train_set.store_dataset()
            val_set = OpenFaceDataset(dataset_name="validation_set", working_dir=self.working_dir,
                                      file_name=self.file_name,
                                      data_instances=val_instances)
            val_set.store_dataset()
            test_set = OpenFaceDataset(dataset_name="test_set", working_dir=self.working_dir, file_name=self.file_name,
                                       data_instances=test_instances)
            test_set.store_dataset()
        else:
            return train_instances, val_instances, test_instances

    def get_instances(self, ids):
        return [instance for instance in self.instances if instance.pt_id in ids]

    def compute_statistics(self, trial_id_stats=None, return_output=False):
        # Count data
        print("Number of patients:", len(self.ids))
        print("Number of data instances:", self.len)

        # Count trial types
        n_stimulus = len([instance for instance in self.instances if instance.trial_type == 1])
        n_control = self.len - n_stimulus
        OpenFaceDataset.draw_pie_bar_plot([n_control, n_stimulus], self.trial_types, "Trial types",
                                          self.preliminary_dir + "trial_types_pie")
        OpenFaceDataset.draw_pie_bar_plot([n_control, n_stimulus], self.trial_types, "Trial types",
                                          self.preliminary_dir + "trial_types_bar", do_bar=True)

        # Count age (at patient level)
        ages = []
        ages_categorical = []
        for pt_id in self.ids:
            for instance in self.instances:
                if instance.pt_id == pt_id:
                    age = instance.age
                    ages.append(age)
                    ages_categorical.append(OpenFaceInstance.categorize_age(age, self.is_boa))
                    break
        maximum = 24 if not self.is_boa else 7
        OpenFaceDataset.draw_hist(ages, maximum, "Age distribution (in months)", self.preliminary_dir +
                                  "age_distr")
        maximum = 3 if not self.is_boa else 2
        OpenFaceDataset.draw_hist(ages_categorical, maximum, "Age distribution (categorical)",
                                  self.preliminary_dir + "age_categ_distr", self.age_groups)

        # Count number of trials
        trials = [instance.trial_id for instance in self.instances]
        maximum = 90 if not self.is_boa else 20
        OpenFaceDataset.draw_hist(trials, maximum, "Trial distribution", self.preliminary_dir + "trial_distr")

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
        ages_categorical_all = [OpenFaceInstance.categorize_age(instance.age, self.is_boa)
                                for instance in self.instances]
        OpenFaceDataset.interaction_count(ages_categorical_all, trials_categorical, self.age_groups,
                                          self.trial_id_groups, "Age (categorical)", "Trial ID (categorical)",
                                          self.preliminary_dir + "age_vs_trial.png")

        # Count sequence duration
        durations = [instance.face_info.shape[0] / OpenFaceDataset.fc for instance in self.instances]
        print("Mean sequence duration = ", str(np.mean(durations)) + "s", "(std = " + str(np.std(durations)) + ")")
        threshold = 300 if not self.is_boa else 250
        for instance in self.instances:
            length = instance.face_info.shape[0]
            if (not self.is_toy or self.is_boa) and length < threshold:
                print("Patient", instance.pt_id, "(trial " + str(instance.trial_id) + ") has", length, "data points")

        if return_output:
            return ages_categorical, ages_categorical_all, trials_categorical

    def remove_short_sequences(self):
        removable = []
        threshold = 300 if not self.is_boa else 250
        for instance in self.instances:
            if instance.face_info.shape[0] < threshold:
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
            x, y, extra_info = self.__getitem__(idx)
            return x, y, extra_info

    @staticmethod
    def interaction_count(var_list1, var_list2, var_labels1, var_labels2, var_name1, var_name2, output_path):
        pairs = list(zip(var_list1, var_list2))
        pair_counts = Counter(pairs)
        l1 = len(var_labels1)
        l2 = len(var_labels2)
        cm = np.zeros((l1, l2), dtype=int)
        for k, v in pair_counts.items():
            cm[k[0], k[1]] = int(v)
        plt.figure(figsize=(8, 4))
        plt.imshow(cm, cmap="Reds")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                plt.text(j, i, f"{val.item()}", ha="center", va="center", color="black", fontsize="xx-large")
        font_size = 10 if l2 < 10 else 8
        plt.xticks(list(range(l2)), var_labels2, rotation=10, fontsize=font_size)
        plt.xlabel(var_name2)
        plt.yticks(list(range(l1)), var_labels1)
        plt.ylabel(var_name1)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

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
            if len(labels) > 2:
                labels = [label.upper() for label in labels]
                plt.xticks(rotation=20, fontsize=6)
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
            divisor = 3 if maximum > 20 else 0.5
            bins = int(maximum / divisor)
        else:
            bins = maximum
        ax = plt.hist(data, bins=bins)

        if labels is None:
            plt.xticks(range(0, maximum + 1, 4))
        else:
            plt.xticks([], [])
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
    def load_dataset(working_dir, dataset_name, task_type=None, train_trial_id_stats=None, is_boa=False, is_toy=False,
                     s3=None):
        if is_toy and not is_boa:
            data_fold = "data/toy/"
        elif is_boa:
            data_fold = "data/boa/"
        else:
            data_fold = OpenFaceDataset.data_fold
        file_path = working_dir + data_fold + dataset_name + ".pt"
        file = open(file_path, "rb") if s3 is None else s3.open(file_path, "rb")
        dataset = pickle.load(file)
        dataset.task_type = task_type

        if train_trial_id_stats is not None:
            dataset.trial_id_stats = train_trial_id_stats

        return dataset

    @staticmethod
    def get_subjective_trial_stats(data):
        trial_stats_dict = {}
        for pt_id in data.ids:
            pt_trials = [instance.trial_id for instance in data.instances if instance.pt_id == pt_id]
            m_trial = np.mean(pt_trials)
            s_trial = np.std(pt_trials)
            trial_stats_dict.update({pt_id: (m_trial, s_trial)})

        return trial_stats_dict


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
