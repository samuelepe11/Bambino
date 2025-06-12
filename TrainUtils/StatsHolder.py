# Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu


# Class
class StatsHolder:

    # Define class attributes
    eps = 1e-7
    table_stats = ["f1", "auc", "mcc", "fnr", "fpr"]
    comparable_stats = {"acc": "Accuracy", "f1": "F1-score", "auc": "AUC", "mcc": "MCC", "fnr": "FNR",
                        "fpr": "FPR"}

    def __init__(self, loss, acc, tp, tn, fp, fn, auc, extra_stats=None, desired_class=None,
                 get_distribution_params=False, alpha_ci=0.05):
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
            (self.n_vals, self.sens, self.spec, self.precis, self.neg_pred_val, self.f1, self.mcc, self.auc, self.fnr,
             self.fpr) = extra_stats
        else:
            self.n_vals = 1
            self.sens = tp / (tp + fn + self.eps)
            self.spec = tn / (tn + fp + self.eps)
            self.precis = tp / (tp + fp + self.eps)
            self.neg_pred_val = tn / (tn + fn + self.eps)
            self.f1 = 2 * self.sens * self.precis / (self.sens + self.precis + self.eps)
            self.mcc = (tp * tn - fp * fn) / np.sqrt(np.float64(tp + fp) * (fn + tn) * (tp + fn) * (fp + tn) + self.eps)
            self.fnr = fn / (tp + fn + self.eps)
            self.fpr = fp / (tn + fp + self.eps)

        # Compute the Macro-Averaged statistics for the multiclass scenario and compute general purpose statistics
        if get_distribution_params:
            # Get means
            self.acc_m = np.mean(self.acc)
            self.auc_m = np.mean(self.auc)
            self.sens_m = np.mean(self.sens)
            self.spec_m = np.mean(self.spec)
            self.precis_m = np.mean(self.precis)
            self.neg_pred_val_m = np.mean(self.neg_pred_val)
            self.f1_m = np.mean(self.f1)
            self.mcc_m = np.mean(self.mcc)
            self.fnr_m = np.mean(self.fnr)
            self.fpr_m = np.mean(self.fpr)

            # Get standard deviations
            self.acc_s = np.std(self.acc)
            self.auc_s = np.std(self.auc)
            self.sens_s = np.std(self.sens)
            self.spec_s = np.std(self.spec)
            self.precis_s = np.std(self.precis)
            self.neg_pred_val_s = np.std(self.neg_pred_val)
            self.f1_s = np.std(self.f1)
            self.mcc_s = np.std(self.mcc)
            self.fnr_s = np.std(self.fnr)
            self.fpr_s = np.std(self.fpr)

            # Get 95% confidence intervals
            percentiles = [100 * alpha_ci / 2, 100 * 1 - alpha_ci / 2]
            self.acc_ci = np.percentile(self.acc, percentiles)
            self.auc_ci = np.percentile(self.auc, percentiles)
            self.sens_ci = np.percentile(self.sens, percentiles)
            self.spec_ci = np.percentile(self.spec, percentiles)
            self.precis_ci = np.percentile(self.precis, percentiles)
            self.neg_pred_val_ci = np.percentile(self.neg_pred_val, percentiles)
            self.f1_ci = np.percentile(self.f1, percentiles)
            self.mcc_ci = np.percentile(self.mcc, percentiles)
            self.fnr_ci = np.percentile(self.fnr, percentiles)
            self.fpr_ci = np.percentile(self.fpr, percentiles)

            if self.loss[0] is not None:
                self.loss_m = np.mean(self.loss)
                self.loss_s = np.std(self.loss)
                self.loss_ci = np.percentile(self.loss, percentiles)

        if isinstance(self.f1, np.ndarray) and not get_distribution_params:
            if desired_class is not None:
                self.target_auc = self.auc[desired_class]
                self.target_sens = self.sens[desired_class]
                self.target_spec = self.spec[desired_class]
                self.target_precis = self.precis[desired_class]
                self.target_neg_pred_val = self.neg_pred_val[desired_class]
                self.target_f1 = self.f1[desired_class]
                self.target_mcc = self.mcc[desired_class]
                self.target_fnr = self.fnr[desired_class]
                self.target_fpr = self.fpr[desired_class]

            self.auc = np.mean(self.auc)
            self.sens = np.mean(self.sens)
            self.spec = np.mean(self.spec)
            self.precis = np.mean(self.precis)
            self.neg_pred_val = np.mean(self.neg_pred_val)
            self.f1 = np.mean(self.f1)
            self.mcc = np.mean(self.mcc)
            self.fnr = np.mean(self.fnr)
            self.fpr = np.mean(self.fpr)

        self.calibration_results = None

    @staticmethod
    def average_bootstrap_values(stat_list, alpha_ci=0.05):
        n_vals = len(stat_list)
        loss = [stat_list[i].loss for i in range(n_vals)]
        acc = [stat_list[i].acc for i in range(n_vals)]
        sens = [stat_list[i].target_sens for i in range(n_vals)]
        spec = [stat_list[i].target_spec for i in range(n_vals)]
        precis = [stat_list[i].target_precis for i in range(n_vals)]
        neg_pred_val = [stat_list[i].target_neg_pred_val for i in range(n_vals)]
        f1 = [stat_list[i].target_f1 for i in range(n_vals)]
        mcc = [stat_list[i].target_mcc for i in range(n_vals)]
        auc = [stat_list[i].target_auc for i in range(n_vals)]
        fnr = [stat_list[i].target_fnr for i in range(n_vals)]
        fpr = [stat_list[i].target_fpr for i in range(n_vals)]

        extra_stats = (n_vals, sens, spec, precis, neg_pred_val, f1, mcc, auc, fnr, fpr)
        stats = StatsHolder(loss, acc, None, None, None, None, None, extra_stats=extra_stats,
                            get_distribution_params=True, alpha_ci=alpha_ci)
        return stats

    @staticmethod
    def draw_compare_plots(stat_list, extra_feature_name, set_type, path, desired_stats=None, alpha_ci=0.05):
        fig_size = (10, 5)
        if desired_stats is None:
            all_stats = True
            desired_stats = StatsHolder.comparable_stats.keys()
        else:
            all_stats = False
        desired_stats_names = [StatsHolder.comparable_stats[s] for s in desired_stats]

        # Transform data into DataFrames
        dfs = []
        value_vars = [StatsHolder.comparable_stats[s] for s in desired_stats]
        if "-" not in extra_feature_name:
            extra_var = extra_feature_name.upper()
        else:
            extra_var = extra_feature_name.split(" - ")
            extra_var = extra_var[-1].upper()
        extra_var += " CODE"
        missing = []
        for i in range(len(stat_list)):
            stats = stat_list[i]
            if stats is not None:
                d = dict((StatsHolder.comparable_stats[k], stats.__dict__[k]) for k in desired_stats)
                df = pd.DataFrame(data=d)
                df = pd.melt(df, value_vars=value_vars, var_name="Metric type", value_name="Value")
                df[extra_var] = i
                dfs.append(df)
            else:
                missing.append(i)

        df = pd.concat(dfs)
        if len(missing) > 0:
            for m in missing:
                df1 = df.copy()
                # df1 = df1[df1[extra_var] == 0]
                df1.loc[:, "Value"] = 0.
                df1.loc[:, extra_var] = m
                df = pd.concat([df, df1])

        # Rename variables
        if not all_stats:
            for i in range(len(desired_stats)):
                df["Metric type"] = np.where(df["Metric type"].eq(desired_stats[i]), desired_stats_names[i],
                                             df["Metric type"])

        # Draw bar plot
        plt.figure(figsize=fig_size)
        n_hue_levels = df[extra_var].nunique()
        if desired_stats is not None and len(desired_stats) == 1:
            ax = sns.barplot(x=extra_var, y="Value", data=df.dropna(), width=1,
                             errorbar=("ci", 100 * (1 - alpha_ci)), err_kws={"linewidth": 1}, capsize=0.2)
            plt.xticks(range(0, np.max(df[extra_var]), 5))
        else:
            ax = sns.barplot(x="Metric type", y="Value", data=df, hue=extra_var, width=0.5,
                             errorbar=("ci", 100 * (1 - alpha_ci)), err_kws={"linewidth": 1.5}, capsize=0.3,
                             palette=sns.color_palette("dark:#5A9_r", n_colors=n_hue_levels))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.ylim([0, 1])
        if desired_stats is not None and len(desired_stats) == 1:
            init = StatsHolder.comparable_stats[desired_stats[0]]
        else:
            init = "Statistics"
        plt.title(init + " on the " + set_type.value.upper() + " set")
        plt.ylabel("Metric value with 95% CI")

        if all_stats:
            addon = ""
        else:
            addon = "_".join(desired_stats) + "_"
        title_start = path + addon
        plt.savefig(title_start + set_type.value + "_compare_barplot.jpg", dpi=300)
        plt.close()

        # Draw boxplot
        plt.figure(figsize=fig_size)
        if desired_stats is not None and len(desired_stats) == 1:
            ax = sns.boxplot(x=extra_var, y="Value", data=df.dropna())
            plt.xticks(range(0, np.max(df[extra_var]), 5))
        else:
            ax = sns.boxplot(x="Metric type", y="Value", data=df, hue=extra_var,
                             palette=sns.color_palette("dark:#5A9_r", n_colors=n_hue_levels))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.ylim([-0.05, 1.05])
        plt.title(init + " on the " + set_type.value.upper() + " set")
        plt.ylabel("Metric value")

        plt.savefig(title_start + set_type.value + "_compare_boxplot.jpg", dpi=300)
        plt.close()

        # Draw violin plot
        plt.figure(figsize=fig_size)
        if desired_stats is not None and len(desired_stats) == 1:
            ax = sns.violinplot(x=extra_var, y="Value", data=df.dropna())
            plt.xticks(range(0, np.max(df[extra_var]), 5))
        else:
            ax = sns.violinplot(x="Metric type", y="Value", data=df, hue=extra_var,
                                palette=sns.color_palette("dark:#5A9_r", n_colors=n_hue_levels))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.ylim([-0.05, 1.05])
        plt.title(init + " on the " + set_type.value.upper() + " set")
        plt.ylabel("Metric value")

        plt.savefig(title_start + set_type.value + "_compare_violinplot.jpg", dpi=300)
        plt.close()

    @staticmethod
    def statistically_compare_stats(stat_list, extra_feature_name, set_type, path, checking_patterns=None,
                                    alpha=0.05, n_rep=100):
        if "histograms" not in os.listdir(path):
            os.mkdir(path + "histograms/")
        path += "histograms/"
        print("\nStatistical comparison for performances in different " + extra_feature_name.upper() + " subgroups:")

        for stat in StatsHolder.comparable_stats.keys():
            stat_name = StatsHolder.comparable_stats[stat]
            vals = []
            for i in range(len(stat_list)):
                stat_elem = stat_list[i]
                if stat_elem is not None:
                    vals.append(stat_elem.__dict__[stat])
                else:
                    vals.append(list(np.zeros(n_rep)))

            # Assess normality
            plt.figure(figsize=(10, 5))
            plt.xlabel(stat_name)
            plt.ylabel("Absolute frequency")
            plt.title(stat_name + " in the " + set_type.value + " set")
            normality = []
            for i in range(len(vals)):
                val = vals[i]
                label = extra_feature_name + " = " + str(i)
                plt.hist(val, label=label, alpha=0.5, bins=100)
                normality.append(StatsHolder.verify_normality(stat_name + " for " + extra_feature_name + " = " + str(i),
                                                              val, alpha))
            plt.legend()
            plt.savefig(path + extra_feature_name + "_" + set_type.value + "_" + stat + "_histogram.jpg", dpi=300)
            plt.close()

            for ind1, ind2 in checking_patterns:
                val1 = vals[ind1]
                normality1 = normality[ind1]
                val2 = vals[ind2]
                normality2 = normality[ind2]

                # Compare variances
                same_var = StatsHolder.compare_variance(stat_name + " (" + extra_feature_name + " = " + str(ind1) +
                                                        " vs. " + extra_feature_name + " = " + str(ind2) + ")", val1,
                                                        val2, [normality1, normality2], alpha)

                # Compare means
                _ = StatsHolder.compare_mean(stat_name + " (" + extra_feature_name + " = " + str(ind1) + " vs. " +
                                             extra_feature_name + " = " + str(ind2) + ")", val1, val2,
                                            [normality1, normality2], same_var, alpha)

    @staticmethod
    def verify_normality(stat_descr, values, alpha):
        if len(np.unique(values)) == 1:
            # Avoid issues owed to identical stats values
            p_value = 1.0
        else:
            _, p_value = shapiro(values)

        normality = p_value > alpha
        if normality:
            addon = ""
        else:
            addon = " NOT"
        print("   > " + stat_descr + " is" + addon + " normally distributed (p-value: {:.2e}".format(p_value) + ")")

        return normality

    @staticmethod
    def compare_variance(stat_descr, arr1, arr2, normality, alpha):
        if len(np.unique(arr1)) == 1 and len(np.unique(arr2)) == 1 and arr1[0] == arr2[0]:
            # Avoid issues owed to identical stats values
            p_value = 1.0
        else:
            if normality[0] and normality[1]:
                # Levene test
                _, p_value = levene(arr1, arr2, center="mean")
            else:
                # Brown–Forsythe test
                _, p_value = levene(arr1, arr2, center="median")

        same_var = p_value > alpha
        if same_var:
            addon = ""
        else:
            addon = " DON'T"
        print("     > The " + stat_descr + " samples" + addon + " have the same variance (p-value: {:.2e}".format(p_value) +
              ")")

        return same_var

    @staticmethod
    def compare_mean(stat_descr, arr1, arr2, normality, same_variance, alpha, equality_check=False):
        if equality_check:
            alternative = "two-sided"
        else:
            alternative = "greater"

        if len(np.unique(arr1)) == 1 and len(np.unique(arr2)) == 1 and arr1[0] == arr2[0]:
            # Avoid issues owed to identical stats values
            p_value = 1.0
        else:
            if normality[0] and normality[1]:
                # T-test if same variance, Welch's T-test (analog to T-test with Satterhwaite method) otherwise
                results = ttest_ind(arr1, arr2, equal_var=same_variance, alternative=alternative)
                p_value = results.pvalue
                if np.isnan(p_value):
                    p_value = 1.0
            else:
                # Mann-Whitney U rank test
                _, p_value = mannwhitneyu(arr1, arr2, alternative=alternative, method="auto")

        h = p_value < alpha
        if not equality_check:
            # Call from the main function
            m1_wins = h
            if m1_wins:
                addon = ""
            else:
                addon = " NOT"
            print("       > The former " + stat_descr + " sample is" + addon + " better than the latter (p-value: {:.2e}".format(p_value) + ")")

            if not m1_wins:
                StatsHolder.compare_mean(stat_descr, arr1, arr2, normality, same_variance, alpha, equality_check=True)

        else:
            # Call from another compare_mean method for equality check
            equal = not h
            if equal:
                addon = ""
            else:
                addon = " NOT"
            print("       > The " + stat_descr + " samples are" + addon + " equal (p-value: {:.2e}".format(p_value) + ")")

        return h
