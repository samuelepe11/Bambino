# Import packages
import numpy as np
from scipy.stats import norm


# Class
class StatsHolder:

    # Define class attributes
    eps = 1e-7
    table_stats = ["f1", "auc", "mcc", "precis", "neg_pred_val"]

    def __init__(self, loss, acc, tp, tn, fp, fn, auc, extra_stats=None, desired_class=None):
        # Initialize attributes
        self.loss = loss
        self.acc = acc
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.auc = auc

        # Compute extra stats
        if extra_stats is not None:
            self.n_vals, self.sens, self.spec, self.precis, self.neg_pred_val, self.f1, self.mcc, self.auc = extra_stats
        else:
            self.n_vals = 1
            self.sens = tp / (tp + fn + self.eps)
            self.spec = tn / (tn + fp + self.eps)
            self.precis = tp / (tp + fp + self.eps)
            self.neg_pred_val = tn / (tn + fn + self.eps)
            self.f1 = 2 * self.sens * self.precis / (self.sens + self.precis + self.eps)
            self.mcc = (tp * tn - fp * fn) / np.sqrt(np.float64(tp + fp) * (fn + tn) * (tp + fn) * (fp + tn) + self.eps)

        # Compute the Macro-Averaged statistics for the multiclass scenario
        if isinstance(self.f1, np.ndarray):
            if desired_class is not None:
                self.target_auc = self.auc[desired_class]
                self.target_sens = self.sens[desired_class]
                self.target_spec = self.spec[desired_class]
                self.target_precis = self.precis[desired_class]
                self.target_neg_pred_val = self.neg_pred_val[desired_class]
                self.target_f1 = self.f1[desired_class]
                self.target_mcc = self.mcc[desired_class]

            self.auc = np.mean(self.auc)
            self.sens = np.mean(self.sens)
            self.spec = np.mean(self.spec)
            self.precis = np.mean(self.precis)
            self.neg_pred_val = np.mean(self.neg_pred_val)
            self.f1 = np.mean(self.f1)
            self.mcc = np.mean(self.mcc)

        self.calibration_results = None

    @staticmethod
    def compute_ci(phat, n_vals, ci_alpha):
        z = norm.ppf(1 - ci_alpha / 2, loc=0, scale=1)
        delta = z * np.sqrt(phat * (1 - phat) / n_vals)

        p_min = phat - delta
        if p_min < 0:
            p_min = 0
        p_max = phat + delta
        if p_max > 1:
            p_max = 1

        return [p_min, p_max]
