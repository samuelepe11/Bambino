# Import packages
import numpy as np
from scipy.stats import norm


# Class
class StatsHolder:

    # Define class attributes
    eps = 1e-7

    def __init__(self, loss, acc, tp, tn, fp, fn, extra_stats=None):
        # Initialize attributes
        self.loss = loss
        self.acc = acc
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

        # Compute extra stats
        if extra_stats is not None:
            self.n_vals, self.sens, self.spec, self.precis, self.f1, self.mcc = extra_stats
        else:
            self.n_vals = 1
            self.sens = tp / (tp + fn + self.eps)
            self.spec = tn / (tn + fp + self.eps)
            self.precis = tp / (tp + fp + self.eps)
            self.f1 = 2 * self.sens * self.precis / (self.sens + self.precis + self.eps)
            self.mcc = (tp * tn - fp * fn) / np.sqrt(np.float64(tp + fp) * (fn + tn) * (tp + fn) * (fp + tn) + self.eps)

        # Compute the Macro-Averaged statistics for the multiclass scenario
        if isinstance(self.f1, np.ndarray):
            self.sens = np.mean(self.sens)
            self.spec = np.mean(self.spec)
            self.precis = np.mean(self.precis)
            self.f1 = np.mean(self.f1)
            self.mcc = np.mean(self.mcc)

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
