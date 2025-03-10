# Import packages
import random
import numpy as np
from collections import Counter

from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.OpenFaceInstance import OpenFaceInstance


# Class
class BoaOpenFaceDataset(OpenFaceDataset):
    # Define class attributes
    age_groups = ["[3-5.5)", "[5.5-7]"]
    sex_groups = ["Male", "Female"]
    speaker_groups = ["Left", "Right"]

    # Define folders
    data_fold = "data/boa/"
    results_fold = "results/boa/"

    def __init__(self, dataset_name, working_dir, file_name, data_instances=None, audio_groups=None):
        super().__init__(dataset_name, working_dir, file_name, data_instances, is_boa=True)

        # Define class attributes on the entire dataset
        if audio_groups is not None:
            self.audio_groups = audio_groups
        else:
            self.audio_groups = np.unique([instance.audio for instance in self.instances])

    def __getitem__(self, idx):
        x, y, extra = super().__getitem__(idx)
        age, trial_id_categorical, trial_id = extra

        instance = self.instances[idx]
        sex = OpenFaceDataset.preprocess_label(instance.sex)
        audio = list(self.audio_groups).index(instance.audio)
        audio = OpenFaceDataset.preprocess_label(audio)
        speaker = instance.speaker if instance.speaker is not None else -1
        speaker = OpenFaceDataset.preprocess_label(speaker)

        return x, y, [age, trial_id_categorical, trial_id, sex, audio, speaker]

    def split_dataset(self, train_perc, is_boa=False):
        # Get set-specific instances
        train_instances, val_instances, test_instances = super().split_dataset(train_perc, is_boa=True)

        # Create and store datasets
        train_set = BoaOpenFaceDataset(dataset_name="training_set", working_dir=self.working_dir,
                                       file_name=self.file_name, data_instances=train_instances,
                                       audio_groups=self.audio_groups)
        train_set.store_dataset()
        val_set = BoaOpenFaceDataset(dataset_name="validation_set", working_dir=self.working_dir,
                                     file_name=self.file_name, data_instances=val_instances,
                                     audio_groups=self.audio_groups)
        val_set.store_dataset()
        test_set = BoaOpenFaceDataset(dataset_name="test_set", working_dir=self.working_dir, file_name=self.file_name,
                                      data_instances=test_instances, audio_groups=self.audio_groups)
        test_set.store_dataset()

    def compute_statistics(self, trial_id_stats=None, return_output=False):
        ages_categorical, ages_categorical_all, trials_categorical = super().compute_statistics(trial_id_stats,
                                                                                                True)

        # Count gender (at patient level)
        sexes = []
        for pt_id in self.ids:
            for instance in self.instances:
                if instance.pt_id == pt_id:
                    sex = instance.sex
                    sexes.append(sex)
                    break
        OpenFaceDataset.draw_hist(sexes, 2, "Gender distribution", self.preliminary_dir + "sex_distr",
                                  self.sex_groups)

        # Count audios
        audio_dict = {a: 0 for a in self.audio_groups}
        for instance in self.instances:
            audio_dict[instance.audio] += 1
        OpenFaceDataset.draw_pie_bar_plot(np.array(list(audio_dict.values())), self.audio_groups,
                                          "Audio distribution", self.preliminary_dir + "audio_distr", do_bar=True)

        # Count speakers
        speakers = [instance.speaker for instance in self.instances if instance.speaker is not None]
        OpenFaceDataset.draw_hist(speakers, len(self.speaker_groups), "Speaker distribution", self.preliminary_dir
                                  + "speaker_distr", self.speaker_groups)

        # Count instances by both age and gender (at patient level)
        OpenFaceDataset.interaction_count(ages_categorical, sexes, self.age_groups, self.sex_groups,
                                          "Age (categorical)", "Gender",
                                          self.preliminary_dir + "age_vs_gender.png")

        # Count instances by both age and speaker
        OpenFaceDataset.interaction_count(ages_categorical_all, speakers, self.age_groups, self.speaker_groups,
                                          "Age (categorical)", "Speaker",
                                          self.preliminary_dir + "age_vs_speaker.png")

        # Count instances by both age and sound
        audios = [list(self.audio_groups).index(instance.audio) for instance in self.instances]
        OpenFaceDataset.interaction_count(ages_categorical_all, audios, self.age_groups, self.audio_groups,
                                          "Age (categorical)", "Sound",
                                          self.preliminary_dir + "age_vs_audio.png")

        # Count instances by both gender and number of trials
        sexes_all = [instance.sex for instance in self.instances]
        OpenFaceDataset.interaction_count(sexes_all, trials_categorical, self.sex_groups, self.trial_id_groups,
                                          "Gender", "Trial ID (categorical)",
                                          self.preliminary_dir + "sex_vs_trial.png")

        # Count instances by both gender and speaker
        OpenFaceDataset.interaction_count(sexes_all, speakers, self.sex_groups, self.speaker_groups, "Gender",
                                          "Speaker", self.preliminary_dir + "sex_vs_speaker.png")

        # Count instances by both gender and sound
        OpenFaceDataset.interaction_count(sexes_all, audios, self.sex_groups, self.audio_groups, "Gender",
                                          "Sound", self.preliminary_dir + "sex_vs_audio.png")

        # Count instances by both speaker and number of trials
        OpenFaceDataset.interaction_count(speakers, trials_categorical, self.speaker_groups, self.trial_id_groups,
                                          "Speaker", "Trial ID (categorical)",
                                          self.preliminary_dir + "speaker_vs_trial.png")

        # Count instances by both speaker and sound
        OpenFaceDataset.interaction_count(speakers, audios, self.speaker_groups, self.audio_groups, "Speaker",
                                          "Sound", self.preliminary_dir + "speaker_vs_audio.png")

        # Count instances by both sound and number of trials
        OpenFaceDataset.interaction_count(trials_categorical, audios, self.trial_id_groups, self.audio_groups,
                                          "Trial ID (categorical)", "Sound",
                                          self.preliminary_dir + "audio_vs_trial.png")


# Main
if __name__ == "__main__":
    # Define seeds
    seed = 1
    random.seed(seed)

    # Define variables
    working_dir1 = "./../../"
    file_name1 = "processed_openface_boa_data.csv"
    dataset_name1 = "complete_dataset"

    # Read data
    dataset1 = BoaOpenFaceDataset(dataset_name=dataset_name1, working_dir=working_dir1, file_name=file_name1)

    # Compute statistics
    dataset1.compute_statistics()

    # Remove short sequences
    # dataset1.remove_short_sequences()

    # Divide dataset
    train_perc1 = 0.7
    # dataset1.split_dataset(train_perc=train_perc1)

    # Load training set
    print()
    train_set1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="training_set", is_boa=True)
    train_set1.compute_statistics()
    # train_set1.store_dataset()

    # Load training set
    print()
    val_set1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="validation_set", is_boa=True)
    val_set1.compute_statistics(train_set1.trial_id_stats)

    # Load training set
    print()
    test_set1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="test_set", is_boa=True)
    test_set1.compute_statistics(train_set1.trial_id_stats)

