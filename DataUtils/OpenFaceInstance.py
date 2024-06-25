# Import packages
import numpy as np


# Class
class OpenFaceInstance:

    # Define class attributes
    dim_dict = {"g": 8, "h": 13, "f": 17}
    dim_names = {"g": "Gaze direction", "h": "Head pose", "f": "Facial expression"}
    dim_labels = {"g": ["gaze_0_x", "gaze_1_x", "gaze_angle_x", "gaze_0_y", "gaze_1_y", "gaze_angle_y", "gaze_0_z",
                        "gaze_1_z"],
                  "h": ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Ry_smooth", "pose_Rz", "p_scale",
                        "p_rx", "p_ry", "p_rz", "p_tx", "p_ty"],
                  "f": ["au01", "au02", "au04", "au05", "au06", "au07", "au09", "au10", "au12", "au14", "au15", "au17",
                        "au20", "au23", "au25", "au26", "au45"]}

    def __init__(self, trial_data):
        trial_data = trial_data.to_numpy()

        # Read fixed attributes
        self.pt_id = trial_data[0, 0]
        self.trial_id = trial_data[0, 7]
        self.age = trial_data[0, 3]
        self.sex = trial_data[0, 4]

        self.clinician_pred = trial_data[0, 16]
        self.trial_type = trial_data[0, 8]

        # Read time varying attributes
        self.gaze_info = trial_data[:, 20:28].astype(np.float32)
        self.head_info = trial_data[:, 28:41].astype(np.float32)
        self.face_info = trial_data[:, 41:58].astype(np.float32)

    @staticmethod
    def categorize_age(age):
        age = np.round(age)
        if 7 <= age <= 11:
            age_categorical = 0
        elif 12 <= age <= 18:
            age_categorical = 1
        elif 19 <= age <= 24:
            age_categorical = 2
        else:
            age_categorical = None

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
