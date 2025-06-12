# Import packages
import random
import numpy as np
import copy

from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.OpenFaceInstance import OpenFaceInstance


# Class
class ToyOpenFaceDataset(OpenFaceDataset):
    # Define class attributes
    sex_groups = ["Female", "Male"]
    time_threshold = 4.5

    # Define folders
    data_fold = "data/toy/"
    results_fold = "results/toy/"

    def __init__(self, dataset_name, working_dir, file_name, data_instances=None, is_boa=False):
        super().__init__(dataset_name, working_dir, file_name, data_instances, is_boa, is_toy=True)

    def __getitem__(self, idx):
        x, y, extra = super().__getitem__(idx)
        age, trial_id_categorical, trial_id = extra

        instance = self.instances[idx]
        sex = OpenFaceDataset.preprocess_label(instance.sex)

        return x, y, [age, trial_id_categorical, trial_id, sex]

    def adjust_data_length(self):
        threshold = int(self.time_threshold * OpenFaceDataset.fc)
        size_samples = self.show_durations(True)

        tmp = []
        for instance in self.instances:
            instance_len = instance.face_info.shape[0]
            if instance_len > threshold:
                rand_size = random.choice(size_samples)
                instance_tmp = copy.copy(instance)
                instance_tmp.gaze_info = instance.gaze_info[:rand_size, :]
                instance_tmp.head_info = instance.head_info[:rand_size, :]
                instance_tmp.face_info = instance.face_info[:rand_size, :]
                tmp.append(instance_tmp)

                ToyOpenFaceDataset.adjust_remaining(tmp, instance, instance_len, rand_size, size_samples, threshold,
                                                    instance.trial_id)
            else:
                tmp.append(instance)

        self.instances = tmp
        self.len = len(self.instances)

        print("\nAfter preprocessing...")
        self.show_durations()

    def show_durations(self, get_parameters=False):
        size_samples = [instance.face_info.shape[0] for instance in self.instances if instance.trial_type and
                        instance.clinician_pred]
        true_trial_durations = [sample / OpenFaceDataset.fc for sample in size_samples]
        print("Correctly classified stimulus trials: mean sequence duration =", str(np.mean(true_trial_durations)) + "s",
              "(std = " + str(np.std(true_trial_durations)) + ", n = " + str(len(true_trial_durations)) + ")")

        false_trial_durations = [instance.face_info.shape[0] / OpenFaceDataset.fc for instance in self.instances
                                 if instance.trial_type and not instance.clinician_pred]
        print("Wrongly classified stimulus trials: mean sequence duration =", str(np.mean(false_trial_durations)) + "s",
              "(std = " + str(np.std(false_trial_durations)) + ", n = " + str(len(false_trial_durations)) + ")")

        control_durations = [instance.face_info.shape[0] / OpenFaceDataset.fc for instance in self.instances
                             if not instance.trial_type]
        print("Control trials: mean sequence duration =", str(np.mean(control_durations)) + "s", "(std = " +
              str(np.std(control_durations)) + ", n = " + str(len(control_durations)) + ")")

        if get_parameters:
            return size_samples

    def split_dataset(self, train_perc, is_child_dataset=False):
        # Get set-specific instances
        train_instances, val_instances, test_instances = super().split_dataset(train_perc, is_child_dataset=True)

        if not is_child_dataset:
            # Create and store datasets
            train_set = ToyOpenFaceDataset(dataset_name="training_set", working_dir=self.working_dir,
                                           file_name=self.file_name, data_instances=train_instances)
            train_set.store_dataset()
            val_set = ToyOpenFaceDataset(dataset_name="validation_set", working_dir=self.working_dir,
                                         file_name=self.file_name, data_instances=val_instances)
            val_set.store_dataset()
            test_set = ToyOpenFaceDataset(dataset_name="test_set", working_dir=self.working_dir, file_name=self.file_name,
                                          data_instances=test_instances)
            test_set.store_dataset()
        else:
            return train_instances, val_instances, test_instances

    def compute_statistics(self, trial_id_stats=None, return_output=False):
        ages_categorical, ages_categorical_all, trials_categorical = super().compute_statistics(trial_id_stats,
                                                                                                True)

        # Count sex (at patient level)
        sexes = []
        for pt_id in self.ids:
            for instance in self.instances:
                if instance.pt_id == pt_id:
                    sex = instance.sex
                    sexes.append(sex)
                    break
        OpenFaceDataset.draw_hist(sexes, 2, "Sex distribution", self.preliminary_dir + "sex_distr",
                                  self.sex_groups)

        # Count instances by both age and sex (at patient level)
        OpenFaceDataset.interaction_count(ages_categorical, sexes, self.age_groups, self.sex_groups,
                                          "Age (categorical)", "Sex",
                                          self.preliminary_dir + "age_vs_gender.png")

        # Count instances by both sex and number of trials
        sexes_all = [instance.sex for instance in self.instances]
        OpenFaceDataset.interaction_count(sexes_all, trials_categorical, self.sex_groups, self.trial_id_groups,
                                          "Sex", "Trial ID (categorical)",
                                          self.preliminary_dir + "sex_vs_trial.png")

        if return_output:
            return ages_categorical, ages_categorical_all, trials_categorical, sexes, sexes_all

    @staticmethod
    def adjust_remaining(tmp, instance, instance_len, rand_size_old, size_samples, threshold, old_trial_id):
        if not instance.trial_type and instance_len - rand_size_old > np.min(size_samples):
            if instance_len - rand_size_old >= threshold:
                rand_size_new = rand_size_old + random.choice(size_samples)
                instance_tmp = copy.copy(instance)
                instance_tmp.gaze_info = instance.gaze_info[rand_size_old:rand_size_new, :]
                instance_tmp.head_info = instance.head_info[rand_size_old:rand_size_new, :]
                instance_tmp.face_info = instance.face_info[rand_size_old:rand_size_new, :]
                new_trial_id = old_trial_id + 0.1
                instance_tmp.trial_id = new_trial_id
                tmp.append(instance_tmp)
                ToyOpenFaceDataset.adjust_remaining(tmp, instance, instance_len, rand_size_new, size_samples, threshold,
                                                    new_trial_id)


# Main
if __name__ == "__main__":
    # Define seeds
    seed = 1
    random.seed(seed)

    # Define variables
    working_dir1 = "./../../"
    file_name1 = "processed_openface_toy_data.csv"

    # Read data
    dataset1 = ToyOpenFaceDataset(dataset_name="complete_dataset_before_adjust", working_dir=working_dir1,
                                  file_name=file_name1)

    # Compute statistics
    dataset1.compute_statistics()

    # Reduce control length
    print("-----------------------------------------------------------------------------------------------------------")
    dataset1.adjust_data_length()
    dataset_name1 = "complete_dataset"
    dataset1.dataset_name = dataset_name1
    dataset1.preliminary_dir = dataset1.results_dir + dataset1.preliminary_fold + dataset_name1 + "/"

    # Compute statistics
    print("-----------------------------------------------------------------------------------------------------------")
    dataset1.compute_statistics()

    # Divide dataset
    train_perc1 = 0.7
    dataset1.split_dataset(train_perc=train_perc1)

    # Load training set
    print()
    train_set1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="training_set", is_toy=True)
    train_set1.compute_statistics()
    train_set1.store_dataset()

    # Load training set
    print()
    val_set1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="validation_set", is_toy=True)
    val_set1.compute_statistics(train_set1.trial_id_stats)

    # Load training set
    print()
    test_set1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="test_set", is_toy=True)
    test_set1.compute_statistics(train_set1.trial_id_stats)
