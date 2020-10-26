import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from read_data.utils import *
from read_data.resource import *
from read_data.perf import *
from analyze_interference import *


def rescalePerf(exp_series, df):
    for t1 in df["t1"].unique():
        metric = exp_series.getPerfMetricsForTypeShort(t1)
        avg_metric = f"avg_{metric}"
        std_metric = f"std_{metric}"
        factor = df.loc[(df["t1"] == t1) & (df["ai_no"] == 1) & (df["tasks"] == 1), avg_metric].mean()
        for t2 in df.loc[df["t1"] == t1, "t2"].unique():
            selected_rows = (df["t1"] == t1) & (df["t2"] == t2) & (df["ai_no"] == 1)
            df.loc[selected_rows, f"{avg_metric}_rescaled"] = df.loc[selected_rows, avg_metric] / factor
            df.loc[selected_rows, f"{std_metric}_rescaled"] = df.loc[selected_rows, std_metric] / factor


def getPerfVsCpu(cpu_res, perf_res):
    result = {}
    result.update(cpu_res)
    result.update(perf_res)
    result["perf_vs_cpu"] = result["cpu_agg"].merge(result["perf_agg"], on=["expid", "t1", "t2", "tasks"], how="inner")
    return result


def printPerfVsCpuMultipleSeries(exp_series_list, cpu_limit=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = [ax for axs1 in axs for ax in axs1]
    tasks = exp_series_list[0].tasks

    for i, t1 in enumerate(tasks):
        axs[i].set_title(f"performance of {t1}")
        for t2 in tasks:
            xs, ys, ys_err = getTrainingDataMultipleSeries(exp_series_list, t1, t2, inverse_throughput_y=True, \
                                             x_col="avg_cpu", cpu_limit=cpu_limit)
            axs[i].scatter(xs, ys, label=t2)
        axs[i].legend()
    plt.show()


def printPerfVsCpu(exp_series, cpu_limit=None):
    printPerfVsCpuMultipleSeries([exp_series], cpu_limit)


def readPerfVsCpu(exp_series, silent=True, skip_cpu=False):
    print(f"perf vs cpu {exp_series.path}")
    perf_res = getPerfDataAll(exp_series)
    if skip_cpu:  # Experimental ad-hoc
        result = {}
        result.update(perf_res)
        result["perf_vs_cpu"] = result["perf_agg"]
    else:
        cpu_res = getCpuDataAll(exp_series)
        result = getPerfVsCpu(cpu_res, perf_res)
    rescalePerf(exp_series, result["perf_vs_cpu"])
    exp_series.dfs = result
    exp_series.df = result["perf_vs_cpu"]
    if not silent:
        printPerfVsCpu(exp_series, result["perf_vs_cpu"])


def getResourceSinglePod(expid, role, ts, resource="cpu", ai="ai-1"):
    tmin, tmax = ts
    paths = getResDataPaths(expid, 0, role, resource)
    res = readResData(paths[0])
    return res.loc[dfInterval(res, tmin, tmax), ["datetime", "value"]]


def getResourceLimitsRecord(ai_type, ai_role, resource, df):
    result = {"type": ai_type, "role": ai_role, "resource": resource}
    if resource == "cpu":
        unit = "m"  # milicores
    elif resource == "memory":
        df["value"] = df["value"] / 2 ** 20
        unit = "Mi"
    else:
        raise ValueError(f"Unsupported resource type {resource}")
    values = df["value"]
    result.update({"avg": values.mean(), "std": values.std(), "min": values.min(), "max": values.max()})
    result.update({"unit": unit})
    return result


def getResourceLimits(exp_series):
    results = pd.DataFrame()
    for ai_type in exp_series.tasks:
        try:
            expid = exp_series.getExperiment(ai_type, ai_type)
        except KeyError:
            print(f"ERROR: Experiment for a type pair {ai_type},{ai_type} not found. Skipping.")
            continue
        print(expid.expid)
        df = readExp(expid)
        if "ai_2" in df["ai_name"].unique():
            tmax = df.loc[df["ai_name"] == "ai_2", "datetime"].min()
        else:
            tmax = df.loc[df["ai_name"] == "ai_1", "datetime"].max()
        tmin = df.loc[df["ai_name"] == "ai_1", "datetime"].min()
        ts = (tmin, tmax)
        for role in ai_info.AI_TYPE_TO_ROLE[ai_type]:
            for resource in ["cpu", "memory"]:
                df = getResourceSinglePod(expid, role, ts, resource)
                record = getResourceLimitsRecord(ai_type, role, resource, df)
                results = results.append(toSingleRowDF(record), ignore_index=True)
    return results


def appendResourceLimits(exp_series, input="resource.csv", output="resource2.csv"):
    limits = pd.read_csv(input)
    new_limits = getResourceLimits(exp_series)
    limits = limits.append(new_limits, ignore_index=True)
    limits.to_csv(output)


def getMetricsStatsColumnNames(quantiles=(0.25, 0.5, 0.75)):
    results = []
    for m in ["throughput", "latency"]:
        results += [f"{m}_samples_count", f"avg_{m}", f"std_{m}"]
        for q in quantiles:
            results.append(formatQuantileColumnName(m, q))
    return results


def updateRowWithMetricsIfExists(df, exp_series, t1, t2, tasks, metric, d):
    try:
        expid = exp_series.getExperiment(t1, t2)
    except KeyError:
        return d
    values = df.loc[
        (df["t1"] == t1) & (df["t2"] == t2) & (df["ai_no"] == 1) & (df["tasks"] == tasks), metric].to_numpy()
    if values.size > 1:
        raise ValueError(f"Got unexpected number of values {values.size} for " \
                         f"{t1} {t2} {tasks} {metric}")
    elif values.size == 1:
        d.update({f"{t1}_{t2}": values[0]})
    return d


# cols: [expid, metric], exp1 exp2 exp3, ...
def getMetricStatComparisonDF(exp_series, exp_series_data_dicts, tasks, max_task_count):
    results = pd.DataFrame()
    for metric in getMetricsStatsColumnNames():
        for i, exp_series1 in enumerate(exp_series):
            df = exp_series_data_dicts[i]["perf_vs_cpu"]
            for task_no in range(1, max_task_count + 1):
                d = {"exp_series": exp_series1.name, "metric": metric, "ai_no": 1, "tasks": task_no}
                for t1 in tasks:
                    for t2 in tasks:
                        d = updateRowWithMetricsIfExists(df, exp_series1, t1, t2, task_no, metric, d)
                results.append(toSingleRowDF(d), ignore_index=True)
    return results


def getMSEForData(xs, ys):
    reg = linear_model.LinearRegression()
    reg.fit(xs, ys)
    error = mean_squared_error(ys, reg.predict(xs))
    return error, xs.size


# Regression level = all, t1, t1, t2
def computeMSE(exp_series, x_label, regression_level, cpu_limit=None):
    df = exp_series.df
    if regression_level not in ["all", "t1", "t2"]:
        raise ValueError(f"Unsupported value of regression level {regression_level}")
    errors_and_weights = []
    t1s = df["t1"].unique() if regression_level in ["t1", "t2"] else [None]
    for t1 in t1s:
        t2s = df.loc[df["t1"] == t1, "t2"].unique() if regression_level == "t2" else [None]
        for t2 in t2s:
            xs, ys, _ = getTrainingData(exp_series, t1, t2, x_label, \
                inverse_throughput_y=True, cpu_limit=cpu_limit)
            errors_and_weights.append(getMSEForData(xs, ys))
    errors, weights = zip(*errors_and_weights)
    return np.average(errors, weights=weights)


def formatInterferenceMatrixInGoFormat(exp_series_list):
    def formatMatrix(matrix):
        res = "CoefficientsMatrix{"
        for row in matrix:
            res += "{" + ",".join([str(v) for v in row]) + "},\n"
        res += "}"
        return res
    for exp_series in exp_series_list:
        matrix = extractInterferenceMatrix(exp_series)
        print(f"\"{exp_series.node}\": " + formatMatrix(matrix) + ",")
