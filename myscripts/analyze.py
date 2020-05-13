import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


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
    def __init__(self, path, tasks, ai_role_count={}):
        self.tasks = tasks
        self.path = path
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
                    print(f"Missing record for {t1} {t2}")
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
        self.ai_role_count.update(ai_role_count)

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


def readAppData(path, skiprows=28):
    df = pd.read_csv(path, skiprows=skiprows, parse_dates=True)
    df["datetime"] = pd.to_datetime(df["time_h"])
    return df


def readResData(path):
    res = pd.read_csv(path)
    res["datetime"] = pd.to_datetime(res["time"], utc=True)
    return res


def readExp(expid):
    paths = glob.glob(os.path.join(expid.path, "VM_runtime_app_*.csv"))
    if len(paths) != 1:
        raise ValueError(f"Unexpeted number of performance data files {len(paths)} != 1 for " \
                         f"{expid.path}")
    return readAppData(paths[0])


def dfInterval(df, tmin, tmax):
    return (tmin <= df["datetime"]) & (df["datetime"] <= tmax)


def getSplitIntervals(df):
    ais = df["ai_name"].unique()
    tss = [df.loc[df["ai_name"] == ai, "datetime"].min() for ai in ais]
    tss.append(df["datetime"].max())
    tss.sort()
    return list(zip(tss[:-1], tss[1:]))


def splitDF(df, timestampDF=None, ai_name="ai_1"):
    if timestampDF is None:
        timestampDF = df
    tss = getSplitIntervals(timestampDF)
    if ai_name is not None:
        df = df[df["ai_name"] == ai_name]
    return [df[dfInterval(df, *ts)] for ts in tss]


def getTrainingData(exp_series, df, t1=None, t2=None, x_col="avg_cpu", inverse_throughput_y=False, cpu_limit=None):
    if not t1:
        results = [[], [], []]
        for t1 in df["t1"].unique():
            result = getTrainingData(exp_series, df, t1, t2, x_col, inverse_throughput_y, cpu_limit)
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
        data = df.loc[selected_rows, [x_col, avg_metric, std_metric]]

        xs = data.loc[selected_rows, x_col].to_numpy()
        xs = xs.reshape(-1, 1)
        ys = data.loc[selected_rows, avg_metric].to_numpy()
        ys_err = data.loc[selected_rows, std_metric].to_numpy()
        if inverse_throughput_y and (metric in ["throughput_rescaled", "bandwith_rescaled"]):
            ys = 1. / ys
        return xs, ys, ys_err


def plotPerf(df, expid, metric_name="app_latency"):
    dfs = splitDF(df)
    fig, ax = plt.subplots()
    metric, metric_stddev = getMetrics(dfs)
    ax.errorbar(range(1, len(metric) + 1), metric, yerr=metric_stddev)
    ax.set_ylabel(metric_name)
    ax.set_title(expid.expid)
    plt.show()


def plotRegressionLine(ax, x, y, yerr, b, expid=None, metric_name="app_latency"):
    x = np.array(x)
    ax.errorbar(x, y, yerr, color="m", fmt="o")
    y_pred = b[0] + b[1] * x
    ax.plot(x, y_pred, color="g")


def getXs(ys):
    return list(range(1, len(ys) + 1))


def getRegressionCoef(reg):
    return np.array([reg.intercept_, reg.coef_[0]])


def analyzeData(expid, df, ax, metric_name="latency"):
    print(expid.expid)
    xs, ys, ys_error = getTrainingData(expid.exp_series, df, expid.t1, expid.t2, "tasks")
    reg = linear_model.LinearRegression()
    reg.fit(xs, ys)
    # error = mean_squared_error(ys, reg.predict(xs))
    coef = np.array([reg.intercept_, reg.coef_[0]])
    plotRegressionLine(ax, xs, ys, ys_error, coef, expid, metric_name)
    return coef


def analyzeInterferenceGrid(exp_series_list, dfs):
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
            metric = exp_series.getPerfMetricsForTypeShort(t1)
            sign = -1. if metric in ["throughput", "bandwith"] else 1.
            formatLegend(ax, t1, t2, metric, i, j)
            for k, exp_series in enumerate(exp_series_list):
                try:
                    expid = exp_series.getExperiment(t1, t2)
                except KeyError:
                    # print(f"WARNING: No experiment data for {t1} {t2}")
                    results[k, i, j] = 0
                else:
                    coefs = analyzeData(expid, dfs[k], ax, metric_name=metric)
                    results[k, i, j] = coefs[1] * sign
    plt.show()
    return results


def getResDataPaths(expid, ai_no, role, resource):
    paths = glob.glob(os.path.join(expid.path, \
                                   f"resources/metric_pod_*{role}-ai-{ai_no+1}_{resource}.csv"))
    expected_count = expid.exp_series.ai_role_count[role]
    if len(paths) != expected_count:
        raise ValueError(f"Unexpected number of resources {len(paths)} != {expected_count} for " \
                         f"{expid.expid} {role} {ai_no+1} {resource}")
    return paths


def toSingleRowDF(d):
    d2 = {}
    for k, v in d.items():
        d2.update({k: pd.Series(v, index=[0])})
    return pd.DataFrame(d2)


# DF: expid, t1, t2, ai_no, ai_role, tasks, avg_cpu, std_cpu, cpu_samples_count
def getResourceDatapoint(expid, ai_no, ai_role, tasks, ts, df):
    df = df.loc[dfInterval(df, *ts), ["datetime", "value"]]
    values = df["value"]
    d = {"expid": expid.expid, "t1": expid.t1, "t2": expid.t2, "ai_no": ai_no + 1}
    d.update({"ai_role": ai_role, "tasks": tasks + 1})
    if values.count() == 0:
        raise ValueError(f"No data samples for {expid.expid} {ai_no+1} {ai_role} {tasks}")
    d.update({"avg_cpu": values.mean(), "std_cpu": values.std(), "cpu_samples_count": values.count()})
    return toSingleRowDF(d)


def getResourceDatapointsSingleFile(expid, ai_no, ai_role, tss, df):
    results = pd.DataFrame()
    for tasks in range(ai_no, len(tss)):
        result = getResourceDatapoint(expid, ai_no, ai_role, tasks, tss[tasks], df)
        results = results.append(result, ignore_index=True)
    return results


def getResourceDatapoints(expid, ai_no, tss):
    results = pd.DataFrame()
    t = expid.t1 if ai_no == 0 else expid.t2
    for ai_role in expid.exp_series.ai_type_role[t]:
        paths = getResDataPaths(expid, ai_no, ai_role, "cpu")
        for path in paths:
            df = readResData(path)
            result = getResourceDatapointsSingleFile(expid, ai_no, ai_role, tss, df)
            results = results.append(result, ignore_index=True)
    return results


# cpu, cpu_aggregate, perf_cpu
def getCpuData(exp_series):
    print("Getting cpu data")
    results = pd.DataFrame()
    for t1 in exp_series.tasks:
        for t2 in exp_series.tasks:
            try:
                expid = exp_series.getExperiment(t1, t2)
            except KeyError:
                print(f"WARNING: No experiment data for {t1} {t2}")
            else:
                df = readExp(expid)
                tss = getSplitIntervals(df)
                max_ais = len(tss)
                for ai_no in range(max_ais):
                    result = getResourceDatapoints(expid, ai_no, tss)
                    results = results.append(result, ignore_index=True)
    return results


def getCpuAggregate(cpu):
    print("Aggregating cpu data")
    cpu_sum = cpu.groupby(["expid", "t1", "t2", "tasks"], as_index=False).sum()
    cpu_sum.pop("ai_no")
    return cpu_sum


def getCpuDataAll(exp_series):
    cpu = getCpuData(exp_series)
    agg = getCpuAggregate(cpu)
    return {"cpu": cpu, "cpu_agg": agg}


def formatQuantileColumnName(m, q):
    return f"{m}_quantile{int(q * 100)}"


def getPerfAggregateForMetric(expid, df, ai_no, tasks, metric, d, quantiles=[0.25, 0.5, 0.75]):
    df = df.loc[df[metric].notna(), :]
    if df.empty:
        msg = f"Performance aggregation failed: no datapoints for " \
              f"{expid.expid} ai_no={ai_no+1} tasks={tasks+1} metric={metric}"
        if ai_no == 0:  # We cannot analyze data without metrics for ai 0
            raise ValueError(msg)
        else:  # We do not care about "noise tasks" performance
            print(f"WARNING: {msg}")
            return d
    perf = df[metric]
    m = metric[len("app_"):]
    d.update({f"avg_{m}": perf.mean(), f"std_{m}": perf.std()})
    d.update({f"{m}_samples_count": perf.count()})
    for q in quantiles:
        d.update({formatQuantileColumnName(m, q): perf.quantile(q)})
    return d


def getPerfAggregateForAIAndTaskNumber(expid, df, ai_no, tasks, ts):
    d = {"expid": expid.expid, "ai_no": ai_no + 1, "tasks": tasks + 1}
    ai_name = f"ai_{ai_no+1}"
    ai_type = expid.t1 if ai_no == 0 else expid.t2
    df = df.loc[dfInterval(df, *ts) & (df["ai_name"] == ai_name), :]
    for metric in expid.exp_series.getPerfMetricsForType(ai_type):
        d = getPerfAggregateForMetric(expid, df, ai_no, tasks, metric, d)
    return toSingleRowDF(d)


# expid, t1, t2, ai_name, tasks
def getPerfAggregate(exp_series, input_df):
    print("Aggregating perf data")
    results = pd.DataFrame()
    for t1 in input_df["t1"].unique():
        for t2 in input_df.loc[input_df["t1"] == t1, "t2"].unique():
            expid = exp_series.getExperiment(t1, t2)
            df = input_df.loc[input_df["expid"] == expid.expid, :]
            tss = getSplitIntervals(df)
            ai_count = len(tss)
            for ai_no in range(ai_count):
                for tasks in range(ai_no, ai_count):
                    result = getPerfAggregateForAIAndTaskNumber(expid, df, ai_no, tasks, tss[tasks])
                    results = results.append(result, ignore_index=True)
    return results


def getPerfData(exp_series):
    print("Getting perf data")
    results = pd.DataFrame()
    for expid in exp_series.type_pair_to_exp.values():
        df = readExp(expid)
        df["expid"] = expid.expid
        df["t1"] = expid.t1
        df["t2"] = expid.t2
        results = results.append(df, ignore_index=True)
    return results


def getPerfDataAll(exp_series):
    perf = getPerfData(exp_series)
    agg = getPerfAggregate(exp_series, perf)
    return {"perf": perf, "perf_agg": agg}


def rescalePerfVsCpu(exp_series, df):
    for t1 in df["t1"].unique():
        metric = exp_series.getPerfMetricsForTypeShort(t1)
        avg_metric = f"avg_{metric}"
        std_metric = f"std_{metric}"
        for t2 in df.loc[df["t1"] == t1, "t2"].unique():
            selected_rows = (df["t1"] == t1) & (df["t2"] == t2) & (df["ai_no"] == 1)
            factor = df.loc[selected_rows & (df["tasks"] == 1), avg_metric].mean()
            df.loc[selected_rows, f"{avg_metric}_rescaled"] = df.loc[selected_rows, avg_metric] / factor
            df.loc[selected_rows, f"{std_metric}_rescaled"] = df.loc[selected_rows, std_metric] / factor


def getPerfVsCpu(cpu_res, perf_res):
    result = {}
    result.update(cpu_res)
    result.update(perf_res)
    result["perf_vs_cpu"] = result["cpu_agg"].merge(result["perf_agg"], on=["expid", "tasks"], how="inner")
    return result


"""
Functions for analyzing performance vs cpu.
"""
#task_to_color = {"redis_ycsb": "green", "wrk": "blue", "linpack": "yellow", "hadoop": "red"}
def printPerfVsCpu(exp_series, df, cpu_limit=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = [ax for axs1 in axs for ax in axs1]

    for i, t1 in enumerate(sorted(df["t1"].unique())):
        axs[i].set_title(f"performance of {t1}")
        for t2 in sorted(df["t2"].unique()):
            xs, ys, ys_err = getTrainingData(exp_series, df, t1, t2, inverse_throughput_y=True, \
                                             x_col="avg_cpu", cpu_limit=cpu_limit)
            axs[i].scatter(xs, ys, label=t2)
        axs[i].legend()
    plt.show()


def analyzePerfVsCpu(exp_series, silent=True):
    print(f"perf vs cpu {exp_series.path}")
    cpu_res = getCpuDataAll(exp_series)
    perf_res = getPerfDataAll(exp_series)
    result = getPerfVsCpu(cpu_res, perf_res)
    rescalePerfVsCpu(exp_series, result["perf_vs_cpu"])
    if not silent:
        printPerfVsCpu(exp_series, result["perf_vs_cpu"])
    return result


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


def getMetrisStatsColumnNames(quantiles=[0.25, 0.5, 0.75]):
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
def computeMSE(df, x_label, regression_level, cpu_limit=None):
    if regression_level not in ["all", "t1", "t2"]:
        raise ValueError(f"Unsupported value of regression level {regression_level}")
    errors_and_weights = []
    t1s = df["t1"].unique() if regression_level in ["t1", "t2"] else [None]
    for t1 in t1s:
        t2s = df.loc[df["t1"] == t1, "t2"].unique() if regression_level == "t2" else [None]
        for t2 in t2s:
            xs, ys, _ = getTrainingData(df, t1, t2, x_label, \
                inverse_throughput_y=True, cpu_limit=cpu_limit)
            errors_and_weights.append(getMSEForData(xs, ys))
    errors, weights = zip(*errors_and_weights)
    return np.average(errors, weights=weights)
