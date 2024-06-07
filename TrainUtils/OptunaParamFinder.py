# Import packages
import optuna
import numpy as np

from TrainUtils.NetworkTrainer import NetworkTrainer
from Types.TaskType import TaskType
from Types.NetType import NetType
from DataUtils.OpenFaceDataset import OpenFaceDataset


# Class
class OptunaParamFinder:

    def __init__(self, model_name, working_dir, task_type, net_type, epochs, batch_size, val_epochs, n_trials):
        self.model_name = model_name
        self.working_dir = working_dir
        self.task_type = task_type
        self.net_type = net_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_epochs = val_epochs

        self.counter = 0
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(),
                                         pruner=optuna.pruners.MedianPruner())
        self.n_trials = n_trials

    def initialize_study(self):
        self.study.optimize(lambda trial: self.objective(trial), self.n_trials)

    def objective(self, trial):
        params = {
            "n_conv_neurons": int(trial.suggest_int("n_conv_neurons", 100, 1500, step=200)),
            "n_conv_layers": 1,
            "kernel_size": 3,
            "hidden_dim": int(2 ** (trial.suggest_int("hidden_dim", 5, 8, step=1))),
            "p_drop": np.round(trial.suggest_float("p_drop", 0, 0.2, step=0.1), 1),
            "n_extra_fc_after_conv": 0,
            "n_extra_fc_final": int(trial.suggest_int("n_extra_fc_final", 0, 3, step=1)),
            "optimizer": "RMSprop",
            "lr": np.round(trial.suggest_float("lr", 0.001, 0.021, step=0.004), 3),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64])
        }

        # Define seeds
        NetworkTrainer.set_seed(111099)

        print("-------------------------------------------------------------------------------------------------------")
        print("Parameters:", params)
        self.counter += 1
        trainer = NetworkTrainer(model_name=self.model_name, working_dir=self.working_dir, task_type=self.task_type,
                                 net_type=self.net_type, epochs=self.epochs, val_epochs=self.val_epochs, params=params)
        val_f1 = trainer.train(show_epochs=False, trial_n=self.counter, trial=trial)
        return val_f1

    def analyze_study(self):
        print("Best study:")
        best_trial = self.study.best_trial
        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))

        fig = optuna.visualization.plot_intermediate_values(self.study)
        fig.show()
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.show()
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        fig.show()
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.show()


if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../../"
    model_name1 = "stimulus_conv2d_optuna"
    net_type1 = NetType.CONV2D
    task_type1 = TaskType.STIM
    epochs1 = 200
    batch_size1 = None
    val_epochs1 = 10

    # Define Optuna model
    n_trials1 = 10
    optuna1 = OptunaParamFinder(model_name=model_name1, working_dir=working_dir1, task_type=task_type1,
                                net_type=net_type1, epochs=epochs1, batch_size=batch_size1,
                                val_epochs=val_epochs1, n_trials=n_trials1)

    # Run search
    optuna1.initialize_study()

    # Evaluate study
    print()
    optuna1.analyze_study()
