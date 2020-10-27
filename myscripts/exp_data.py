import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import ai_info


class ExperimentRecord:
    def __init__(self, t1, t2, path, exp_series):
        self.t1 = t1  # Task type 1
        self.t2 = t2  # Task type 2
        self.path = path
        self.base_path, self.expid = os.path.split(path)
        self.exp_series = exp_series
        self.reg = None
        self.coefs = [0., 0.]
        self.error = np.nan
        self.xs = []
        self.ys = []
        self.ys_error = []


class ExperimentSeries:
    def __init__(self, path, node, config, skip_tasks=()):
        self.type = "linear"
        self.tasks = config.tasks
        self.path = path
        self.node = node
        self.dfs = {}  # Dict with dataframes with resources usage and perf data
        self.df = None  # Main dataframe with aggregated performance vs cpu data
        self.interference_matrix = None
        _, self.name = os.path.split(path)
        self.type_pair_to_exp = dict()
        for t1 in self.tasks:
            for t2 in self.tasks:
                if (t1, t2) in skip_tasks:
                    print(f"Skipping task pair {t1} {t2}")
                    continue
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

        self.ai_role_count = ai_info.AI_ROLE_TO_COUNT.copy()
        self.options = {"interval_boundaries": "first_plus_interval"}
        self.overrideWithConfig(config)

    def overrideWithConfig(self, config):
        if config.ai_role_count:
            self.ai_role_count.update(config.ai_role_count)
        if config.options:
            self.options.update(config.options)

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
        if t1 not in ai_info.AI_TYPE_TO_METRICS:
            raise ValueError("Type not supported")
        if t1 == "linpack":
            return ["app_throughput"]
        if t1 == "filebench":
            return ["app_throughput"]
        if t1 == "netperf":
            return ["app_bandwidth"]
        if t1 == "unixbench":
            return ["app_throughput"]
        if t1 == "multichase":
            return ["app_throughput", "app_errors", "app_completion_time"]
        return ["app_latency", "app_throughput"]

    def getPerfMetricsForTypeShort(self, t1):
        return self.getPerfMetricsForType(t1)[0][len("app_"):]

    def getSplitIntervalMethod(self):
        return self.options["interval_boundaries"]


def getTrainingData(exp_series, t1=None, t2=None, x_col="avg_cpu", inverse_throughput_y=False, cpu_limit=None,
                    tasks_limit=None, subtract_xs=0., subtract_ys=0.):
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
        if tasks_limit:
            selected_rows = selected_rows & (df["tasks"] <= tasks_limit)
        data = df.loc[selected_rows, :]
        if data.empty:
            return np.array([]), np.array([]), np.array([])

        xs = data.loc[selected_rows, x_col].to_numpy()
        xs = xs.reshape(-1, 1)
        xs = xs.astype(np.float64)
        xs -= subtract_xs
        ys = data.loc[selected_rows, avg_metric].to_numpy()
        ys_err = data.loc[selected_rows, std_metric].to_numpy()
        if inverse_throughput_y and (metric in ["throughput_rescaled", "bandwidth_rescaled"]):
            ys = 1. / ys
        ys -= subtract_ys
        return xs, ys, ys_err


def getTrainingDataMultipleSeries(exp_series_list, t1=None, t2=None, x_col="avg_cpu", inverse_throughput_y=False,
                                  cpu_limit=None, tasks_limit=None):
    results = [np.array([]) for _ in range(3)]
    for exp_series in exp_series_list:
        result = getTrainingData(exp_series, t1, t2, x_col, inverse_throughput_y, cpu_limit, tasks_limit)
        for i in range(len(results)):
            results[i] = np.append(results[i], result[i])
    results[0] = results[0].reshape((-1, 1))
    return results
