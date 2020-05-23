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


def computeInterferenceRegression(exp, metric_name="latency"):
    xs, ys, ys_error = getTrainingData(exp.exp_series, exp.t1, exp.t2, "tasks")
    reg = linear_model.LinearRegression()
    reg.fit(xs, ys)
    error = mean_squared_error(ys, reg.predict(xs))
    coefs = np.array([reg.intercept_, reg.coef_[0]])
    exp.xs, exp.ys, exp.ys_error = xs, ys, ys_error
    exp.reg, exp.coefs, exp.error = reg, coefs, error
    return coefs


def computeInterferenceRegressionGrid(exp_series):
    tasks = exp_series.tasks
    n = len(tasks)
    results = np.zeros((n, n))
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            metric = exp_series.getPerfMetricsForTypeShort(t1)
            sign = -1. if metric in ["throughput", "bandwidth"] else 1.
            try:
                expid = exp_series.getExperiment(t1, t2)
            except KeyError:
                # print(f"WARNING: No experiment data for {t1} {t2}")
                results[i, j] = 0
            else:
                coefs = computeInterferenceRegression(expid, metric_name=metric)
                results[i, j] = coefs[1] * sign
    return results


def printInterferenceGridMultipleSeries(exp_series_list, skip_tasks=()):
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
    plt.show()


def printInterferenceGrid(exp_series_list, skip_tasks=()):
    printInterferenceGridMultipleSeries([exp_series_list], skip_tasks)


def analyzeInterferenceGridMultipleSeries(exp_series_list, skip_tasks=()):
    for exp_series in exp_series_list:
        computeInterferenceRegressionGrid(exp_series)
    printInterferenceGridMultipleSeries(exp_series_list, skip_tasks)


def analyzeInterferenceGrid(exp_series, skip_tasks=()):
    return analyzeInterferenceGridMultipleSeries([exp_series], skip_tasks)


def extractInterferenceMatrix(exp_series):
    n = len(exp_series.tasks)
    result = np.zeros((n, n))
    for i, t1 in enumerate(exp_series.tasks):
        for j, t2 in enumerate(exp_series.tasks):
            metric = exp_series.getPerfMetricsForTypeShort(t1)
            sign = -1. if metric in ["throughput", "bandwidth"] else 1.
            try:
                exp = exp_series.getExperiment(t1, t2)
            except KeyError:
                # print(f"WARNING: No experiment data for {t1} {t2}")
                result[i, j] = 0
            else:
                result[i, j] = exp.coefs[1] * sign
    return result
