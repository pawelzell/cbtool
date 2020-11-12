import numpy as np
import os
import datetime

import ai_info
import global_config


class ExperimentConfig:
    # Supported options:
    # "interval_boundaries": "first_plus_interval", "max"
    def __init__(self, tasks, ai_role_count=None, options=None):
        self.tasks = tasks
        self.ai_role_count = ai_role_count
        self.options = options


class SchedulerExperimentConfig:
    def __init__(self, config, rescale_map, node_to_coeffs):
        self.tasks = config.tasks
        self.ai_role_count = config.ai_role_count
        self.options = config.options
        self.rescale_map = rescale_map
        self.node_to_coeffs = node_to_coeffs

    @staticmethod
    def createConfig(config, exp_series_list):
        rescale_map = getRescaleFactorMap(exp_series_list)
        node_to_coeffs = extractNodeToCoeffs(exp_series_list)
        return SchedulerExperimentConfig(config, rescale_map, node_to_coeffs)


class ExperimentTrace:
    # Abstraction for parsed trace.
    def __init__(self, types, intervals):
        self.types = types
        self.intervals = intervals

    @staticmethod
    def parseTrace(path, t1, t2):
        name = t1 if t1 == t2 else f"{t1}_{t2}"
        path = os.path.join(path, name)
        types, intervals = [], []
        with open(path, "r") as f:
            for line in f:
                ExperimentTrace.updateTypesIntervals(types, intervals, line)
        return ExperimentTrace(types, intervals)

    @staticmethod
    def updateTypesIntervals(types, intervals, line):
        lt, li = len(types), len(intervals)
        line = line.strip()
        if line.startswith("aiattach"):
            types.append(line.split(" ")[1])
        elif line.startswith("waitfor"):
            interval = line.split(" ")[1]
            if interval[-1] != "m":
                raise ValueError(f"Interval unit {interval[-1]} not supported {interval}")
            interval = int(interval[:-1])
            if lt == li:
                intervals[-1] += interval
            else:
                intervals.append(interval)
        ExperimentTrace.assertCorrectState(lt, li)


    @staticmethod
    def assertCorrectState(lt, li):
        if lt != li and lt != li+1:
            raise ValueError(f"Incorrect state: (types len, intervals len) ({lt}, {li})")


class RegressionRecord:
    def __init__(self, t1, t2, xs, ys, ys_error, reg, coefs, error):
        self.t1 = t1
        self.t2 = t2
        self.xs = np.array(xs)
        self.ys = ys
        self.ys_error = ys_error
        self.reg = reg
        self.coefs = coefs
        self.error = error

    def plot(self, ax):
        ax.errorbar(self.xs, self.ys, self.ys_error, color="m", fmt="o")
        y_pred = self.coefs[0] + self.coefs[1] * self.xs
        ax.plot(self.xs, y_pred, color="g")


class SchedulerConfigurationWriter:
    def __init__(self, path, section_separator="-"):
        self.path = os.path.join(path, "scheduler_config")
        self.section_separator = section_separator

    @staticmethod
    def createWithDefaults():
        return SchedulerConfigurationWriter(global_config.SCHEDULER_CONFIG_DIR)

    def writeConfiguration(self, exp_series_list, description):
        self.checkSameTasks(exp_series_list)
        t_creation = datetime.datetime.now()
        with open(self.path, "w") as f:
            self.writeHeaderSection(t_creation, exp_series_list, f)
            self.writeTypeListSection(f, exp_series_list)
            self.writeDescription(description, f)
            for exp_series in exp_series_list:
                self.writeInterferenceMatrix(exp_series, f)
            self.writeSectionEnd(f)

        print(f"Configuration written to file {self.path}")

    def writeDescription(self, desc, f):
        f.write(f"description: {desc}\n")

    def writeHeaderSection(self, t_create, exp_series_list, f):
        f.write(f"# Created: {t_create.isoformat()}\n")
        for exp_series in exp_series_list:
            f.write(f"# exp_series: {exp_series.node} {exp_series.path}\n")
        self.writeSectionEnd(f)

    def writeTypeListSection(self, f, exp_series_list):
        f.write(" ".join(exp_series_list[0].tasks))
        f.write("\n")
        self.writeSectionEnd(f)

    def writeSectionEnd(self, f):
        f.write(f"{self.section_separator}\n")

    def writeInterferenceMatrix(self, exp_series, f):
        f.write(f"{exp_series.node}\n")
        for row in exp_series.interference_matrix:
            f.write(" ".join([str(n) for n in row]))
            f.write("\n")

    def checkSameTasks(self, exp_series_list):
        if not exp_series_list:
            raise ValueError(f"Empty exp_series_list not supported")
        expected_tasks = tuple(exp_series_list[0].tasks)
        for exp_series in exp_series_list:
            actual_tasks = tuple(exp_series.tasks)
            if actual_tasks != expected_tasks:
                raise ValueError("Exp series tasks mismatch: {exp_series.node} {actual_tasks} vs {expected_tasks}")


def getRescaleFactorMap(exp_series_list):
    results = {}
    for exp_series in exp_series_list:
        df = exp_series.dfs["perf_agg"]
        result = {}
        for t1 in df["t1"].unique():
            result[t1] = {}
            for metric in ai_info.AI_TYPE_TO_METRICS[t1]:
                avg_metric = f"avg_{metric}"
                agg_df = df.loc[(df["t1"] == t1) & (df["ai_no"] == 1) & (df["tasks"] == 1), avg_metric]
                print(f"Aggregate {exp_series.name} {t1} {metric} {len(agg_df)}")
                result[t1][metric] = agg_df.mean()
        results[exp_series.node] = result
    return results


def extractNodeToCoeffs(exp_series_list):
    results = {}
    for exp_series in exp_series_list:
        results[exp_series.node] = exp_series.interference_matrix
    return results


