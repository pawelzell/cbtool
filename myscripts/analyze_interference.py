import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from exp_data import *


def plotRegressionLine(ax, x, y, yerr, b):
    x = np.array(x)
    ax.errorbar(x, y, yerr, color="m", fmt="o")
    y_pred = b[0] + b[1] * x
    ax.plot(x, y_pred, color="g")


def computeInterferenceRegression(exp, inverse_throughput_y, fit_intercept, task_limit=None):
    subtract_xs = subtract_ys = 0.
    if not fit_intercept:
        subtract_xs = subtract_ys = 1.
    xs, ys, ys_error = getTrainingData(exp.exp_series, exp.t1, exp.t2, "tasks",
                                       inverse_throughput_y=inverse_throughput_y,
                                       x_col_limit=task_limit,
                                       subtract_xs=subtract_xs, subtract_ys=subtract_ys)
    reg = linear_model.LinearRegression(fit_intercept=fit_intercept)
    reg.fit(xs, ys)
    error = mean_squared_error(ys, reg.predict(xs))
    coefs = np.array([reg.intercept_, reg.coef_[0]])
    exp.xs, exp.ys, exp.ys_error = xs, ys, ys_error
    exp.reg, exp.coefs, exp.error = reg, coefs, error
    return coefs


def computeInterferenceRegressionGrid(exp_series, inverse_throughput_y, fit_intercept, task_pair_to_task_limit={}):
    tasks = exp_series.tasks
    n = len(tasks)
    results = np.zeros((n, n))
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            task_limit = task_pair_to_task_limit.get((t1, t2), None)
            try:
                expid = exp_series.getExperiment(t1, t2)
            except KeyError:
                # print(f"WARNING: No experiment data for {t1} {t2}")
                results[i, j] = 0
            else:
                coefs = computeInterferenceRegression(expid, inverse_throughput_y, fit_intercept, task_limit)
                results[i, j] = coefs[1]
    exp_series.interference_matrix = results
    return results


def printInterferenceGridMultipleSeries(exp_series_list, skip_tasks=(), savefig=False):
    tasks = exp_series_list[0].tasks
    n = len(tasks)

    def formatLegend(ax, t1, t2, metric, i, j):
        if not i:
            ax.set_title(f"influence of {t2}")
        if i == n - 1:
            ax.set_xlabel('number of tasks')
        if not j:
            ax.set_ylabel(f"{t1} - {metric}")

    fig, axes = plt.subplots(n, n, figsize=(15., 15.))
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            ax = axes[i, j]
            metric = exp_series_list[0].getPerfMetricsForTypeShort(t1)
            formatLegend(ax, t1, t2, metric, i, j)
            for k, exp_series in enumerate(exp_series_list):
                try:
                    if (t1, t2) in skip_tasks:
                        raise KeyError("Skip task")
                    exp = exp_series.getExperiment(t1, t2)
                except KeyError:
                    pass
                    # print(f"WARNING: No experiment data for {t1} {t2}")
                else:
                    plotRegressionLine(ax, exp.xs, exp.ys, exp.ys_error, exp.coefs)
    if savefig:
        file_name = f"{exp_series_list[0].node}_interference_grid"
        plt.savefig(file_name)
        print(f"Figure saved to {file_name}")
    else:
        plt.show()


def printInterferenceGrid(exp_series_list, skip_tasks=(), savefig=False):
    printInterferenceGridMultipleSeries([exp_series_list], skip_tasks, savefig)


def analyzeInterferenceGridMultipleSeries(exp_series_list, skip_tasks, inverse_throughput_y, fit_intercept,
                                          task_pair_to_task_limit, savefig):
    for exp_series in exp_series_list:
        computeInterferenceRegressionGrid(exp_series, inverse_throughput_y, fit_intercept, task_pair_to_task_limit)
    printInterferenceGridMultipleSeries(exp_series_list, skip_tasks, savefig)


def analyzeInterferenceGrid(exp_series, skip_tasks=(), inverse_throughput_y=True, fit_intercept=False,
                            task_pair_to_task_limit={}, savefig=False):
    return analyzeInterferenceGridMultipleSeries([exp_series], skip_tasks, inverse_throughput_y, fit_intercept, task_pair_to_task_limit, savefig)
