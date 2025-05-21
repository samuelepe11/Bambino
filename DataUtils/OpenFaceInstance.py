# Import packages
import numpy as np
import pandas as pd


# Class
class OpenFaceInstance:

    # Define class attributes
    dim_dict = {"g": 8, "h": 13, "f": 17}
    dim_names = {"g": "Gaze direction", "h": "Head pose", "f": "Facial expression"}
    dim_labels = {"g": ["gaze_0_x", "gaze_1_x", "gaze_angle_x", "gaze_0_y", "gaze_1_y", "gaze_angle_y", "gaze_0_z",
                        "gaze_1_z"],
                  "h": ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Ry_smooth", "pose_Rz", "p_scale",
                        "p_rx", "p_ry", "p_rz", "p_tx", "p_ty"],
                  "f": ["Inner Brow Raiser", "Outer Brow Raiser", "Brow Lowerer", "Upper Lid Raiser", "Cheek Raiser",
                        "Lid Tightener", "Nose Wrinkler", "Upper Lip Raiser", "Lip Corner Puller", "Dimpler",
                        "Lip Corner Depressor", "Chin Raiser", "Lip stretcher", "Lip Tightener", "Lips part",
                        "Jaw Drop", "Blink"]}
    subplot_settings = {"g": [15, 10, 2, 4], "h": [15, 10, 3, 5], "f": [15, 10, 4, 5]}

    def __init__(self, trial_data, is_boa, is_toy):
        trial_data = trial_data.to_numpy()
        if not is_boa:
            pt_ind = 0
            trial_ind = 7
            age_ind = 3
            sex_ind = 4
            trial_type_ind = 8
            if not is_toy:
                gaze_ind = range(20, 28)
                head_ind = range(28, 41)
                face_ind = range(41, 58)
                self.clinician_pred = trial_data[0, 16]
            else:
                gaze_ind = range(21, 29)
                head_ind = range(29, 42)
                face_ind = range(42, 59)
                self.clinician_pred = trial_data[0, 17]

        else:
            pt_ind = 0
            trial_ind = 4
            age_ind = 3
            sex_ind = 1
            trial_type_ind = 5
            gaze_ind = range(17, 25)
            head_ind = range(25, 38)
            face_ind = range(38, 55)

            self.audio = trial_data[0, 6][:-4].replace("_", " ")
            self.speaker = trial_data[0, 7]
            if self.speaker == "left":
                self.speaker = 0
            elif self.speaker == "right":
                self.speaker = 1
            else:
                self.speaker = None

        # Read fixed attributes
        self.pt_id = trial_data[0, pt_ind]
        self.trial_id = trial_data[0, trial_ind]
        self.age = trial_data[0, age_ind]
        self.sex = trial_data[0, sex_ind]
        self.trial_type = trial_data[0, trial_type_ind]

        conf_ind = 19 if not is_boa else 15
        if is_toy:
            # Cut windows from stimulus presentation to reward display
            if not is_boa:
                timestamp = trial_data[:, 10]
                max_time = trial_data[0, 15]
                max_time = max_time if not np.isnan(max_time) else np.inf
                windows_inds = (timestamp >= 0) & (timestamp < max_time)
                trial_data = trial_data[windows_inds, :]

            # Remove unsuccessful acquisitions and interpolate signals
            low_conf_inds = trial_data[:, conf_ind] < 0.5
            trial_data[low_conf_inds, :] = np.nan

        # Read time varying attributes
        self.gaze_info = np.stack([pd.Series(trial_data[:, i].astype(np.float32)).interpolate(method="linear")
                                   for i in gaze_ind], 1)
        self.head_info = np.stack([pd.Series(trial_data[:, i].astype(np.float32)).interpolate(method="linear")
                                   for i in head_ind], 1)
        self.face_info = np.stack([pd.Series(trial_data[:, i].astype(np.float32)).interpolate(method="linear")
                                   for i in face_ind], 1)

    @staticmethod
    def categorize_age(age, is_boa):
        age_categorical = None
        if not is_boa:
            age = np.round(age)
            if 7 <= age <= 11:
                age_categorical = 0
            elif 12 <= age <= 18:
                age_categorical = 1
            elif 19 <= age <= 24:
                age_categorical = 2
        else:
            if 3 <= age < 5.5:
                age_categorical = 0
            elif 5.5 <= age < 7.5:
                age_categorical = 1

        return age_categorical

    @staticmethod
    def categorize_trial_id(trial_id, train_id_stats):
        m_trial, s_trial = train_id_stats
        boundary1 = m_trial - 0.5 * s_trial
        boundary2 = m_trial + 0.5 * s_trial

        if trial_id < boundary1:
            trial_categorical = 0
        elif boundary1 <= trial_id < boundary2:
            trial_categorical = 1
        else:
            # trial_id >= boundary2
            trial_categorical = 2
        return trial_categorical
