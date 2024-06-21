# Import packages
import os
import torch
import torch.nn as nn
import random
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import optuna
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_confusion_matrix
from sklearn.metrics import roc_auc_score
from pandas import DataFrame
from calfram.calibrationframework import select_probability, reliabilityplot, calibrationdiagnosis, classwise_calibration

from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.OpenFaceInstance import OpenFaceInstance
from Types.TaskType import TaskType
from Types.SetType import SetType
from TrainUtils.StatsHolder import StatsHolder
from Types.NetType import NetType
from Networks.StimulusConv1d import StimulusConv1d
from Networks.StimulusConv2d import StimulusConv2d


# Class
class NetworkTrainer:
    # Define class attributes
    convergence_patience = 3
    convergence_thresh = 1e-3

    def __init__(self, model_name, working_dir, task_type, net_type, epochs, val_epochs, params=None,
                 use_cuda=True, separated_inputs=True):
        # Initialize attributes
        self.model_name = model_name
        self.working_dir = working_dir
        self.results_dir = working_dir + OpenFaceDataset.results_fold + OpenFaceDataset.models_fold
        if model_name not in os.listdir(self.results_dir):
            os.mkdir(self.results_dir + model_name)
        self.results_dir += model_name + "/"

        self.task_type = task_type
        self.net_type = net_type
        self.start_time = None
        self.end_time = None

        self.descr_train = None
        self.descr_test = None
        if task_type == TaskType.AGE:
            self.classes = OpenFaceDataset.age_groups
        elif task_type == TaskType.TRIAL:
            self.classes = OpenFaceDataset.trial_id_groups
        else:
            # TaskType.STIM
            self.classes = OpenFaceDataset.trial_types

        if net_type == NetType.CONV1D:
            self.net = StimulusConv1d(params=params, separated_inputs=separated_inputs, n_classes=len(self.classes))
        else:
            # NetType.CONV2D
            self.net = StimulusConv2d(params=params, separated_inputs=separated_inputs, n_classes=len(self.classes))

        # Define training parameters
        self.epochs = epochs
        self.batch_size = self.net.batch_size
        self.val_epochs = val_epochs

        self.criterion = nn.CrossEntropyLoss()
        if params is None:
            self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.net.lr)
        else:
            self.optimizer = getattr(torch.optim, params["optimizer"])(self.net.parameters(), lr=self.net.lr)
        self.train_losses = []
        self.val_losses = []
        self.val_eval_epochs = []
        self.optuna_study = None

        self.use_cuda = torch.cuda.is_available() and use_cuda
        # Otherwise CUDA runs out of memory
        if params is not None and params["n_conv_neurons"] > 1000:
            self.use_cuda = False
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Load datasets
        self.train_data = OpenFaceDataset.load_dataset(working_dir=self.working_dir, dataset_name="training_set",
                                                       task_type=self.task_type)
        self.train_loader, self.train_dim = self.load_data(self.train_data, shuffle=True)

        self.val_data = OpenFaceDataset.load_dataset(working_dir=self.working_dir, dataset_name="validation_set",
                                                     task_type=self.task_type,
                                                     train_trial_id_stats=self.train_data.trial_id_stats)
        self.val_loader, self.val_dim = self.load_data(self.val_data)

        self.test_data = OpenFaceDataset.load_dataset(working_dir=self.working_dir, dataset_name="test_set",
                                                      task_type=self.task_type,
                                                      train_trial_id_stats=self.train_data.trial_id_stats)
        self.test_loader, self.test_dim = self.load_data(self.test_data)

        self.train_mean, self.train_std = self.get_normalization_params(self.train_data)

    def load_data(self, data, shuffle=False):
        dataloader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=shuffle, num_workers=2,
                                collate_fn=NetworkTrainer.custom_collate_fn)
        dim = len(data)

        return dataloader, dim

    def train(self, show_epochs=False, trial_n=None, trial=None):
        if show_epochs:
            self.start_time = time.time()

        if self.use_cuda:
            self.net.set_cuda()
            self.criterion = self.criterion.cuda()
        net = self.net

        if len(self.train_losses) == 0:
            train_stats, val_stats = self.summarize_performance(show_test=False, show_process=False, show_cm=False)
            self.train_losses.append(train_stats.loss)
            self.val_losses.append(val_stats.loss)
            self.val_eval_epochs.append(0)

        for epoch in range(self.epochs):
            net.set_training(True)
            train_loss = 0
            for x, y in self.train_loader:
                loss, _ = self.apply_network(net, x, y)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / len(self.train_data)
            self.train_losses.append(train_loss)

            if epoch % self.val_epochs == 0:
                val_stats = self.test(set_type=SetType.VAL)
                self.val_losses.append(val_stats.loss)
                self.val_eval_epochs.append(epoch + 1)

                if trial is not None:
                    trial.report(val_stats.f1, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            if show_epochs and epoch % 10 == 0:
                print("Epoch " + str(epoch + 1) + "/" + str(self.epochs) + " completed... train loss = " +
                      str(np.round(train_loss, 5)))

            if epoch % self.val_epochs == 0 and len(self.val_losses) > NetworkTrainer.convergence_patience:
                # Check for convergence
                val_mean = np.mean(self.val_losses[-NetworkTrainer.convergence_patience:])
                val_std = np.std(self.val_losses[-NetworkTrainer.convergence_patience:])
                val_cv = val_std / val_mean
                if val_cv < NetworkTrainer.convergence_thresh:
                    print("Validation convergence has been reached sooner...")
                    break

        self.net = net
        if show_epochs:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            print("Execution time:", round(duration / 60, 4), "min")

        self.save_model(trial_n)
        if trial_n is not None:
            _, val_stats = self.summarize_performance(show_test=False, show_process=False, show_cm=False,
                                                      trial_n=trial_n)
            return val_stats.f1

    def apply_network(self, net, x, y):
        x = {key: (x[key] - self.train_mean[key]) / self.train_std[key] for key in x.keys()}
        x = {key: x[key].to(self.device) for key in x.keys()}
        y = y.to(self.device)

        output = net(x)

        # Train loss evaluation
        loss = self.criterion(output, y)
        return loss, output

    def select_dataset(self, set_type):
        if set_type == SetType.TRAIN:
            data = self.train_data
            data_loader = self.train_loader
            dim = self.train_dim
        elif set_type == SetType.VAL:
            data = self.val_data
            data_loader = self.val_loader
            dim = self.val_dim
        else:
            # SetType.TEST
            data = self.test_data
            data_loader = self.test_loader
            dim = self.test_dim

        return data, data_loader, dim

    def compute_stats(self, y_true, y_pred, loss, acc, desired_class=None):
        cm = NetworkTrainer.compute_binary_confusion_matrix(y_true, y_pred, range(len(self.classes)))
        tp = cm[0]
        tn = cm[1]
        fp = cm[2]
        fn = cm[3]
        auc = NetworkTrainer.compute_binary_auc(y_true, y_pred, range(len(self.classes)))

        stats = StatsHolder(loss, acc, tp, tn, fp, fn, auc, desired_class=desired_class)
        return stats

    def test(self, set_type=SetType.TRAIN, show_cm=False, desired_class=None, assess_calibration=False):
        if self.use_cuda:
            self.net.set_cuda()
            self.criterion = self.criterion.cuda()
        net = self.net
        _, data_loader, dim = self.select_dataset(set_type)

        # Store class labels
        y_prob = []
        y_true = []
        y_pred = []
        loss = 0
        net.set_training(False)
        with torch.no_grad():
            for x, y in data_loader:
                temp_loss, output = self.apply_network(net, x, y)
                loss += temp_loss.item()

                # Accuracy evaluation
                prediction = torch.argmax(output, dim=1)

                # Store values for Confusion Matrix calculation
                y_prob.append(output)
                y_true.append(y.to(self.device))
                y_pred.append(prediction)

            y_prob = torch.concat(y_prob)
            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)
            loss /= dim
            acc = torch.sum(y_true == y_pred) / dim
            acc = acc.item()
        stats_holder = self.compute_stats(y_true, y_pred, loss, acc, desired_class=desired_class)

        # Compute multiclass confusion matrix
        cm_name = set_type.value + "_cm"
        if show_cm:
            img_path = self.results_dir + cm_name + ".jpg"
        else:
            img_path = None
        self.__dict__[cm_name] = NetworkTrainer.compute_multiclass_confusion_matrix(y_true, y_pred, self.classes,
                                                                                    img_path)

        if assess_calibration:
            stats_holder.calibration_results = self.assess_calibration(y_true, y_prob, y_pred, set_type)

        return stats_holder

    def assess_calibration(self, y_true, y_prob, y_pred, set_type):
        y_true = y_true.cpu().numpy()
        y_prob = y_prob.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        class_scores = select_probability(y_true, y_prob, y_pred)

        # Store results file
        data = np.concatenate((y_true[:, np.newaxis], y_pred[:, np.newaxis], y_prob), axis=1)
        titles = ["y_true", "y_pred"] + ["y_prob" + str(i) for i in range(y_prob.shape[1])]
        df = DataFrame(data, columns=titles)
        df.to_csv(self.results_dir + set_type.value + "_classification_results.csv", index=False)

        # Draw reliability plot
        reliabilityplot(class_scores, strategy=10, split=False)
        plt.savefig(self.results_dir + set_type.value + "_calibration.png")

        # Compute local metrics
        results, _ = calibrationdiagnosis(class_scores, strategy=10)

        # Compute global metrics
        results_cw = classwise_calibration(results)
        return results_cw

    def save_model(self, trial_n=None):
        if trial_n is None:
            addon = self.model_name
        else:
            addon = "trial_" + str(trial_n - 1)
        file_path = self.results_dir + addon + ".pt"
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
            print("'" + self.model_name + "' has been successfully saved!... train loss: " +
                  str(np.round(self.train_losses[0], 4)) + " -> " + str(np.round(self.train_losses[-1],
                                                                                 4)))

    def summarize_performance(self, show_test=False, show_process=False, show_cm=False, desired_class=None,
                              trial_n=None, assess_calibration=False):
        # Show final losses
        train_stats = self.test(set_type=SetType.TRAIN, show_cm=show_cm, desired_class=desired_class,
                                assess_calibration=assess_calibration)
        print("Training loss = " + str(np.round(train_stats.loss, 5)) + " - Training accuracy = " +
              str(np.round(train_stats.acc * 100, 7)) + "% - Training F1-score = " +
              str(np.round(train_stats.f1 * 100, 7)))
        if desired_class is not None:
            NetworkTrainer.show_performance_table(train_stats, "training")
        if assess_calibration:
            NetworkTrainer.show_calibration_table(train_stats, "training")

        print()
        val_stats = self.test(set_type=SetType.VAL, show_cm=show_cm, desired_class=desired_class,
                              assess_calibration=assess_calibration)
        print("Validation loss = " + str(np.round(val_stats.loss, 5)) + " - Validation accuracy = " +
              str(np.round(val_stats.acc * 100, 7)) + "% - Validation F1-score = " +
              str(np.round(val_stats.f1 * 100, 7)))
        if desired_class is not None:
            NetworkTrainer.show_performance_table(val_stats, "validation")
        if assess_calibration:
            NetworkTrainer.show_calibration_table(val_stats, "training")

        if show_test:
            print()
            test_stats = self.test(set_type=SetType.TEST, show_cm=show_cm, desired_class=desired_class,
                                   assess_calibration=assess_calibration)
            print("Test loss = " + str(np.round(test_stats.loss, 5)) + " - Test accuracy = " +
                  str(np.round(test_stats.acc * 100, 7)) + "% - Test F1-score = " +
                  str(np.round(test_stats.f1 * 100, 7)))
            if desired_class is not None:
                NetworkTrainer.show_performance_table(test_stats, "test")
            if assess_calibration:
                NetworkTrainer.show_calibration_table(test_stats, "training")

        if show_process or trial_n is not None:
            plt.figure()
            plt.plot(self.train_losses, "b", label="Training set")
            plt.plot(self.val_eval_epochs, self.val_losses, "g", label="Validation set")
            plt.legend()
            plt.title("Training curves")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            if show_process:
                plt.show()
            if trial_n is not None:
                plt.savefig(self.results_dir + "trial_" + str(trial_n - 1) + "_curves.jpg")
                plt.close()

        return train_stats, val_stats

    def show_clinician_stim_performance(self, set_type=SetType.TRAIN, desired_class=None):
        data, _, dim = self.select_dataset(set_type)

        if data.clinician_stats is not None:
            # Get true labels and predicted values
            y_true = []
            y_pred = []
            for instance in data.instances:
                yt = torch.Tensor([instance.trial_type])
                y_true.append(yt)

                yp = torch.Tensor([instance.clinician_pred])
                y_pred.append(yp)
            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)

            # Get evaluation metrics
            acc = torch.sum(y_true == y_pred) / dim
            acc = acc.item()
            data.clinician_stats = self.compute_stats(y_true, y_pred, None, acc, desired_class=desired_class)

            # Compute multiclass confusion matrix
            y_true = y_true.to(torch.int64)
            y_pred = y_pred.to(torch.int64)
            img_path = (self.working_dir + OpenFaceDataset.results_fold + "clinician_performance/" + set_type.value
                        + "_cm.jpg")
            data.clinician_cm = NetworkTrainer.compute_multiclass_confusion_matrix(y_true, y_pred,
                                                                                   OpenFaceDataset.trial_types,
                                                                                   img_path)
            data.store_dataset()

        # Show performance
        print(set_type.value + ": Clinician accuracy = " + str(np.round(data.clinician_stats.acc * 100, 7)) +
              "% - Clinician F1-score = " + str(np.round(data.clinician_stats.f1 * 100, 7)) + "%")
        if desired_class is not None:
            NetworkTrainer.show_performance_table(data.clinician_stats, set_type.value)

    def show_model(self):
        print("MODEL:")

        attributes = self.net.__dict__
        for attr in attributes.keys():
            val = attributes[attr]
            if issubclass(type(val), nn.Module):
                print(" > " + attr, "-" * (20 - len(attr)), val)

    @staticmethod
    def compute_binary_confusion_matrix(y_true, y_predicted, classes=None):
        if classes is None:
            # Classical binary computation (class 0 as negative and class 1 as positive)
            tp = torch.sum((y_predicted == 1) & (y_true == 1))
            tn = torch.sum((y_predicted == 0) & (y_true == 0))
            fp = torch.sum((y_predicted == 1) & (y_true == 0))
            fn = torch.sum((y_predicted == 0) & (y_true == 1))

            out = [tp.item(), tn.item(), fp.item(), fn.item()]
            return out
        else:
            # One VS Rest computation for Macro-Averaged F1-score and other metrics
            out = []
            for c in classes:
                y_true_i = (y_true == c).to(int)
                y_predicted_i = (y_predicted == c).to(int)
                out_i = NetworkTrainer.compute_binary_confusion_matrix(y_true_i, y_predicted_i, classes=None)
                out.append(out_i)

            out = np.asarray(out)
            out = [out[:, i] for i in range(out.shape[1])]
            return out

    @staticmethod
    def compute_binary_auc(y_true, y_predicted, classes):
        y_true = y_true.cpu()
        y_predicted = y_predicted.cpu()

        # One VS Rest computation for Macro-Averaged AUC
        out = []
        for c in classes:
            y_true_i = (y_true == c).to(int)
            y_predicted_i = (y_predicted == c).to(int)
            out_i = roc_auc_score(y_true_i, y_predicted_i)
            out.append(out_i)

        out = np.asarray(out)
        return out

    @staticmethod
    def compute_multiclass_confusion_matrix(y_true, y_pred, classes, img_path=None):
        # Compute confusion matrix
        cm = multiclass_confusion_matrix(y_pred, y_true, len(classes))

        # Draw heatmap
        if img_path is not None:
            NetworkTrainer.draw_multiclass_confusion_matrix(cm, classes, img_path)

        return cm

    @staticmethod
    def draw_multiclass_confusion_matrix(cm, labels, img_path):
        plt.figure(figsize=(4, 4))
        cm = cm.cpu()
        plt.imshow(cm, cmap="Reds")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                plt.text(j, i, f"{val.item()}", ha="center", va="center", color="black")
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.xlabel("Predicted class")
        plt.yticks(range(len(labels)), labels, rotation=45)
        plt.ylabel("True class")
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def custom_collate_fn(batch):
        batch_inputs = {key: torch.stack([d[key] for d, _ in batch]) for key in batch[0][0]}
        batch_labels = torch.stack([label for _, label in batch])
        batch_labels = batch_labels.squeeze(1)
        return batch_inputs, batch_labels

    @staticmethod
    def load_model(working_dir, model_name, trial_n=None, use_cuda=True):
        if trial_n is None:
            file_name = model_name
        else:
            file_name = "trial_" + str(trial_n)
        filepath = (working_dir + OpenFaceDataset.results_fold + OpenFaceDataset.models_fold + model_name + "/" +
                    file_name + ".pt")
        with open(filepath, "rb") as file:
            network_trainer = pickle.load(file)

        network_trainer.use_cuda = torch.cuda.is_available() and use_cuda

        # Handle previous versions of the Networks
        if "separated_inputs" not in network_trainer.net.__dict__.keys():
            network_trainer.net.__dict__["separated_inputs"] = True
            network_trainer.net.__dict__["blocks"] = list(OpenFaceInstance.dim_dict.keys())

        # Handle models created with Optuna
        if trial_n is None and network_trainer.model_name.endswith("_optuna"):
            old_model_name = network_trainer.model_name
            network_trainer.model_name = old_model_name[:-7]
            addon = OpenFaceDataset.models_fold
            if addon in network_trainer.results_dir:
                addon = ""
            network_trainer.results_dir = (network_trainer.results_dir[:-(len(old_model_name) + 1)] +
                                           addon + network_trainer.model_name + "/")
        return network_trainer

    @staticmethod
    def get_normalization_params(data):
        loader = DataLoader(dataset=data, batch_size=data.len, shuffle=False, num_workers=0,
                            collate_fn=NetworkTrainer.custom_collate_fn)
        for x, y in loader:
            mean = {key: torch.mean(x[key], dim=(0, 1)) for key in x.keys()}
            std = {key: torch.std(x[key], dim=(0, 1)) for key in x.keys()}
        return mean, std

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.deterministic = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def show_performance_table(stats, set_name):
        print("Performance for", set_name.upper() + " set:")
        for stat in StatsHolder.table_stats:
            print(" - " + stat + ": " + str(np.round(stats.__dict__["target_" + stat] * 100, 2)) + "%")

    @staticmethod
    def show_calibration_table(stats, set_name):
        print("Calibration information for", set_name.upper() + " set:")
        for stat in stats.calibration_results.keys():
            print(" - " + stat + ": " + str(stats.calibration_results[stat]))


# Main
if __name__ == "__main__":
    # Define seed
    NetworkTrainer.set_seed(111099)

    # Define variables
    working_dir1 = "./../../"
    model_name1 = "trial_conv2d"
    net_type1 = NetType.CONV2D
    task_type1 = TaskType.TRIAL
    epochs1 = 100
    trial_n1 = None
    val_epochs1 = 10
    use_cuda1 = True
    separated_inputs1 = True
    assess_calibration1 = False

    # Define trainer
    # params1 = {"n_conv_neurons": 1536, "n_conv_layers": 1, "kernel_size": 7, "hidden_dim": 64, "p_drop": 0.5,
    #            "n_extra_fc_after_conv": 1, "n_extra_fc_final": 1, "optimizer": "RMSprop", "lr": 0.008, "batch_size": 64}  # stimulus_conv1
    # params1 = {"n_conv_neurons": 256, "n_conv_layers": 1, "kernel_size": 3, "hidden_dim": 32, "p_drop": 0.2,
    #            "n_extra_fc_after_conv": 0, "n_extra_fc_final": 1, "optimizer": "RMSprop", "lr": 0.01, "batch_size": 64}  # stimulus_conv2
    params1 = {"n_conv_neurons": 64, "n_conv_layers": 3, "kernel_size": 3, "hidden_dim": 32, "p_drop": 0.1,
               "n_extra_fc_after_conv": 1, "n_extra_fc_final": 1, "optimizer": "RMSprop", "lr": 0.01, "batch_size": 64}  # age_conv1
    trainer1 = NetworkTrainer(model_name=model_name1, working_dir=working_dir1, task_type=task_type1,
                              net_type=net_type1, epochs=epochs1, val_epochs=val_epochs1, params=params1,
                              use_cuda=use_cuda1, separated_inputs=separated_inputs1)

    # Show clinician performance
    # trainer1.show_clinician_stim_performance(set_type=SetType.TRAIN, desired_class=0)
    # trainer1.show_clinician_stim_performance(set_type=SetType.VAL, desired_class=0)

    # Train model
    print()
    trainer1.train(show_epochs=True)
    
    # Evaluate model
    # trainer1 = NetworkTrainer.load_model(working_dir=working_dir1, model_name=model_name1, trial_n=trial_n1,
    #                                      use_cuda=use_cuda1)
    trainer1.summarize_performance(show_test=False, show_process=True, desired_class=0, show_cm=True,
                                   assess_calibration=assess_calibration1)

    # Retrain model
    # trainer1.train(show_epochs=True)
