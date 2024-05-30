# Import packages
import optuna

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
            "n_conv_neurons": int(trial.suggest_categorical("n_conv_neurons", [64, 128, 256])),
            "n_conv_layers": int(trial.suggest_int("n_conv_layers", 1, 3, step=1)),
            "kernel_size": trial.suggest_categorical("kernel_size", [3]),
            "hidden_dim": int(trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256])),
            "p_drop": trial.suggest_float("p_drop", 0.4, 0.6, step=0.1),
            "n_extra_fc_after_conv": int(trial.suggest_int("n_extra_fc_after_conv", 0, 3, step=1)),
            "n_extra_fc_final": int(trial.suggest_int("n_extra_fc_final", 0, 3, step=1)),
            "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop"]),
            "lr": trial.suggest_categorical("lr", [0.001, 0.01, 0.1]),
        }

        # Define seeds
        NetworkTrainer.set_seed(111099)

        print("-------------------------------------------------------------------------------------------------------")
        self.counter += 1
        trainer = NetworkTrainer(model_name=self.model_name, working_dir=self.working_dir, task_type=self.task_type,
                                 net_type=self.net_type, epochs=self.epochs, batch_size=self.batch_size,
                                 val_epochs=self.val_epochs, params=params)
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
    epochs1 = 100
    batch_size1 = 64
    val_epochs1 = 10

    # Define Optuna model
    n_trials1 = 50
    optuna1 = OptunaParamFinder(model_name=model_name1, working_dir=working_dir1, task_type=task_type1,
                                net_type=net_type1, epochs=epochs1, batch_size=batch_size1,
                                val_epochs=val_epochs1, n_trials=n_trials1)

    # Run search
    optuna1.initialize_study()

    # Evaluate study
    print()
    optuna1.analyze_study()
