import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from exp_data import *
from exp_data_utils import RegressionRecord
from read_data.utils import dfTypePair, toSingleRowDF
import global_config


def computeInterferenceRegression(exp_series, t1, t2, inverse_throughput_y, fit_intercept, task_limit=None):
    subtract_xs = subtract_ys = 0.
    if not fit_intercept:
        subtract_xs = subtract_ys = 1.
    xs, ys, ys_error = getTrainingData(exp_series, t1, t2, "tasks",
                                       inverse_throughput_y=inverse_throughput_y,
                                       tasks_limit=task_limit,
                                       subtract_xs=subtract_xs, subtract_ys=subtract_ys)
    reg = linear_model.LinearRegression(fit_intercept=fit_intercept)
    reg.fit(xs, ys)
    error = mean_squared_error(ys, reg.predict(xs))
    coefs = np.array([reg.intercept_, reg.coef_[0]])
    exp_series.type_pair_to_regression[(t1, t2)] = RegressionRecord(t1, t2, xs, ys, ys_error, reg, coefs, error)
    return coefs


def computeInterferenceRegressionGrid(exp_series, inverse_throughput_y, fit_intercept, task_pair_to_task_limit={}):
    tasks = exp_series.tasks
    n = len(tasks)
    results = np.zeros((n, n))
    df = exp_series.df
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            task_limit = task_pair_to_task_limit.get((t1, t2), None)
            if df.loc[dfTypePair(df, t1, t2)].empty:
                results[i, j] = 0
            else:
                coefs = computeInterferenceRegression(exp_series, t1, t2, inverse_throughput_y, fit_intercept,
                                                      task_limit)
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
                if (t1, t2) in skip_tasks:
                    raise KeyError("Skip task")
                try:
                    regression = exp_series.type_pair_to_regression[(t1, t2)]
                except KeyError:
                    print(f"WARNING: No experiment regression record for {t1} {t2}")
                    continue
                regression.plot(ax)
    if savefig:
        file_name = f"interference_grid_{exp_series_list[0].name}"
        file_name = os.path.join(global_config.PLOTS_DIR, file_name)
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
                            tasks_limit=None, savefig=False):
    types = exp_series.tasks
    task_pair_to_task_limit = {(t1, t2): tasks_limit for t1 in types for t2 in types}
    return analyzeInterferenceGridMultipleSeries([exp_series], skip_tasks, inverse_throughput_y, fit_intercept,
                                                 task_pair_to_task_limit, savefig)


def computeExpectedCost(loads, coeffs):
    loads = np.array(loads)
    n = loads.size
    result = []
    for i in range(n):
        cost = 0.
        if loads[i] > 0:
            cost = 1.
            loads[i] -= 1.
            for j in range(n):
                cost += loads[j] * coeffs[i][j]
            loads[i] += 1.
        result.append(cost)
    return result


# DF cols: tasks, ai_no, type, expected_cost
def computeExpectedCostDf(type_list, ai_types, interference_matrix):
    loads = np.zeros(len(ai_types))
    type_to_id = {t: i for i, t in enumerate(ai_types)}
    df = pd.DataFrame()
    d = dict()
    for n_tasks, t in enumerate(type_list, start=1):
        d["tasks"] = n_tasks
        t_id = type_to_id[t]
        loads[t_id] += 1.
        cost_vector = computeExpectedCost(loads, interference_matrix)
        for ai_no in range(1, n_tasks+1):
            d["ai_no"] = ai_no
            d["type"] = type_list[ai_no-1]
            t_id2 = type_to_id[d["type"]]
            d["expected_cost"] = cost_vector[t_id2]
            df = df.append(toSingleRowDF(d), ignore_index=True)
    return df


def plotInterferenceActualVsExpected(exp_series, exp, interference_matrix):
    types_list = exp.trace.types
    cost_df = computeExpectedCostDf(types_list, exp_series.tasks, interference_matrix)
    t = types_list[0]
    select_rows = cost_df["ai_no"] == 1
    expected_df = cost_df.loc[select_rows, ["tasks", "expected_cost"]]

    df = exp_series.df
    select_rows = (df["t1"] == exp.t1) & (df["t2"] == exp.t2) & (df["ai_no"] == 1)
    actual_cost_row = ai_info.getPerfColName(t)
    actual_df = df.loc[select_rows, ["tasks", actual_cost_row]]

    fig, ax = plt.subplots()
    ax.set_title("Performance cost")
    ax.set_xlabel("Number of tasks running")
    ax.scatter(expected_df["tasks"].values, expected_df["expected_cost"].values, label="predicted")
    ax.scatter(actual_df["tasks"].values, actual_df[actual_cost_row].values, label="observed")
    plt.legend()
    plt.show()
