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


class ExperimentRecord:
    def __init__(self, t1, t2, path, exp_series):
        self.t1 = t1  # Task type 1
        self.t2 = t2  # Task type 2
        self.path = path
        self.base_path, self.expid = os.path.split(path)
        self.exp_series = exp_series
        self.reg = None
        self.reg_coef = [0., 0.]
        self.error = np.nan


class ExperimentSeries:
    def __init__(self, path, tasks, ai_role_count=None, options=None):
        self.tasks = tasks
        self.path = path
        self.dfs = {}  # Dict with dataframes with resources usage and perf data
        self.df = None  # Main dataframe with aggregated performance vs cpu data
        _, self.name = os.path.split(path)
        self.type_pair_to_exp = dict()
        for t1 in tasks:
            for t2 in tasks:
                exp_paths = self.getExperimentPaths(t1, t2, path)
                if len(exp_paths) > 1:
                    raise ValueError(f"Found {len(exp_paths)}>1 experiment records for types " \
                                     f"{t1} {t2} in directory {path}")
                elif len(exp_paths) == 1:
                    self.type_pair_to_exp[(t1, t2)] = ExperimentRecord(t1, t2, exp_paths[0], self)
                else:
                    pass
                    #print(f"Missing record for {t1} {t2}")
        print(f"Found {len(self.type_pair_to_exp)} experiment in series {self.name}")

        ai_type_role = dict()
        ai_type_role["redis_ycsb"] = ["ycsb", "redis"]
        ai_type_role["hadoop"] = ["hadoopmaster", "hadoopslave"]
        ai_type_role["linpack"] = ["linpack"]
        ai_type_role["wrk"] = ["wrk", "apache"]
        ai_type_role["filebench"] = ["filebench"]
        ai_type_role["unixbench"] = ["unixbench"]
        ai_type_role["netperf"] = ["netclient", "netserver"]
        self.ai_type_role = ai_type_role

        self.ai_role_count = {}
        for _, ai_roles in self.ai_type_role.items():
            for ai_role in ai_roles:
                self.ai_role_count[ai_role] = 1
        if ai_role_count:
            self.ai_role_count.update(ai_role_count)
        self.options = {"interval_boundaries": "max"}
        if options:
            self.options.update(options)

    @staticmethod
    def getExperimentPaths(t1, t2, base_path):
        expid = t1 if t1 == t2 else f"{t1}_{t2}"

        def matchExpidRegex(e):
            expid_regex = "[0-9]{0,4}" + expid
            i = e.split("/")[-1]
            return bool(re.fullmatch(expid_regex, i))

        pattern = os.path.join(base_path, f"*{expid}")
        expids = glob.glob(pattern)
        return [e for e in expids if matchExpidRegex(e)]

    def getExperiment(self, t1, t2):
        return self.type_pair_to_exp[(t1, t2)]

    def getPerfMetricsForType(self, t1):
        if t1 not in self.tasks:
            raise ValueError("Type not supported")
        if t1 == "linpack":
            return ["app_throughput"]
        if t1 == "filebench":
            return ["app_throughput"]
        if t1 == "netperf":
            return ["app_bandwidth"]
        if t1 == "unixbench":
            return ["app_throughput"]
        return ["app_latency", "app_throughput"]

    def getPerfMetricsForTypeShort(self, t1):
        return self.getPerfMetricsForType(t1)[0][len("app_"):]

    def getSplitIntervalMethod(self):
        return self.options["interval_boundaries"]


def getTrainingData(exp_series, t1=None, t2=None, x_col="avg_cpu", inverse_throughput_y=False, cpu_limit=None):
    df = exp_series.df
    if not t1:
        results = [[], [], []]
        for t1 in df["t1"].unique():
            result = getTrainingData(exp_series, t1, t2, x_col, inverse_throughput_y, cpu_limit)
            for i in range(len(results)):
                results[i] = np.append(results[i], result[i])
        results[0] = results[0].reshape((-1, 1))
        return results
    else:
        metric = f"{exp_series.getPerfMetricsForTypeShort(t1)}_rescaled"
        avg_metric = f"avg_{metric}"
        std_metric = f"std_{metric}"
        selected_rows = (df["t1"] == t1) & (df["ai_no"] == 1)
        if t2:
            selected_rows = selected_rows & (df["t2"] == t2)
        if cpu_limit:
            selected_rows = selected_rows & (df["avg_cpu"] <= cpu_limit)
        data = df.loc[selected_rows, :]
        if data.empty:
            return np.array([]), np.array([]), np.array([])

        xs = data.loc[selected_rows, x_col].to_numpy()
        xs = xs.reshape(-1, 1)
        ys = data.loc[selected_rows, avg_metric].to_numpy()
        ys_err = data.loc[selected_rows, std_metric].to_numpy()
        if inverse_throughput_y and (metric in ["throughput_rescaled", "bandwidth_rescaled"]):
            ys = 1. / ys
        return xs, ys, ys_err


def getTrainingDataMultipleSeries(exp_series_list, t1=None, t2=None, x_col="avg_cpu", inverse_throughput_y=False, cpu_limit=None):
    results = [[], [], []]
    for exp_series in exp_series_list:
        result = getTrainingData(exp_series, t1, t2, x_col, inverse_throughput_y, cpu_limit)
        for i in range(len(results)):
            results[i] = np.append(results[i], result[i])
    results[0] = results[0].reshape((-1, 1))
    return results


def plotRegressionLine(ax, x, y, yerr, b, expid=None, metric_name="app_latency"):
    x = np.array(x)
    ax.errorbar(x, y, yerr, color="m", fmt="o")
    y_pred = b[0] + b[1] * x
    ax.plot(x, y_pred, color="g")


def analyzeData(expid, df, ax, metric_name="latency"):
    print(expid.expid)
    xs, ys, ys_error = getTrainingData(expid.exp_series, expid.t1, expid.t2, "tasks")
    reg = linear_model.LinearRegression()
    reg.fit(xs, ys)
    # error = mean_squared_error(ys, reg.predict(xs))
    coef = np.array([reg.intercept_, reg.coef_[0]])
    plotRegressionLine(ax, xs, ys, ys_error, coef, expid, metric_name)
    return coef


def analyzeInterferenceGrid(exp_series_list, dfs, skip_tasks=[]):
    tasks = exp_series_list[0].tasks
    n = len(tasks)
    results = np.zeros((len(dfs), n, n))

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
            sign = -1. if metric in ["throughput", "bandwith"] else 1.
            formatLegend(ax, t1, t2, metric, i, j)
            for k, exp_series in enumerate(exp_series_list):
                try:
                    if (t1, t2) in skip_tasks:
                        raise KeyError("Skip task")
                    expid = exp_series.getExperiment(t1, t2)
                except KeyError:
                    # print(f"WARNING: No experiment data for {t1} {t2}")
                    results[k, i, j] = 0
                else:
                    coefs = analyzeData(expid, dfs[k], ax, metric_name=metric)
                    results[k, i, j] = coefs[1] * sign
    plt.show()
    return results


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
    result["perf_vs_cpu"] = result["cpu_agg"].merge(result["perf_agg"], on=["expid", "tasks"], how="inner")
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
    for ai_type, roles in exp_series.ai_type_role.items():
        expid = exp_series.getExperiment(ai_type, ai_type)
        print(expid.expid)
        df = readExp(expid)
        tmax = df.loc[df["ai_name"] == "ai_2", "datetime"].min()
        tmin = df.loc[df["ai_name"] == "ai_1", "datetime"].min()
        ts = (tmin, tmax)
        for role in roles:
            for resource in ["cpu", "memory"]:
                df = getResourceSinglePod(expid, role, ts, resource)
                record = getResourceLimitsRecord(ai_type, role, resource, df)
                results = results.append(toSingleRowDF(record), ignore_index=True)
    return results


def getMetricsStatsColumnNames(quantiles=[0.25, 0.5, 0.75]):
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

