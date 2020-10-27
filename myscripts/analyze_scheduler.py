import os
import re
import ai_info
import collections
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from read_data.utils import *
from read_data.perf import *

# from read_data.resource import *
# from analyze_interference import *
#IP_TO_HOST = {"ip_10_2_1_93": "baati", "baati": "baati", "ip_10_2_1_91": "dosa", "dosa": "dosa"}
# TODO
IP_TO_HOST = {"puri.mimuw.edu.pl": "puri.mimuw.edu.pl", "kulcha.mimuw.edu.pl": "kulcha.mimuw.edu.pl"}
METRIC_CBTOOL_PREFIX = "app_"


class SchedulerExperimentRecord:
    def __init__(self, base_path, expid, composition_id, shuffle_id, custom_scheduler, ai_types, exp_series):
        self.base_path = base_path
        self.expid = expid
        self.path = os.path.join(base_path, expid)
        self.composition_id = composition_id
        self.shuffle_id = shuffle_id
        self.custom_scheduler = custom_scheduler
        self.ai_types = ai_types
        self.exp_series = exp_series
        self.split_interval = None

    def getSplitInterval(self, df):
        mins = [df.loc[df["ai_name"] == name, "datetime"].min() for name in df["ai_name"].unique()]
        return max(mins), df["datetime"].max()

    def aggregatePerfForAiNameAndMetric(self, df, d, metric, ai_name):
        df = df.loc[df[metric].notna(), :]
        if df.empty:
            msg = f"Performance aggregation failed: no datapoints for " \
                  f"{d} metric={metric}"
            raise ValueError(msg)
        return getPerfAggregateForMetricHelper(df, d, metric)

    def aggregatePerfForAiName(self, df, ai_name):
        d = {"expid": self.expid, "composition_id": self.composition_id, "shuffle_id": self.shuffle_id,
             "scheduler": self.custom_scheduler, "ai_name": ai_name}
        df = df.loc[(df["ai_name"] == ai_name), :]
        host_names = df["host_name"].unique()
        if len(host_names) != 1:
            raise KeyError(f"Unexpected number of host names for single ai {len(host_names)}!=1")
        d.update({"host_name": host_names[0]})
        ai_roles = df["role"].unique()
        ai_types = [ai_info.AI_ROLE_TO_TYPE[role] for role in ai_roles]
        if len(ai_types) != 1:
            raise KeyError(f"Unexpected number of ai types for single ai {len(ai_types)}!=1")
        d.update({"type": ai_types[0]})
        for m in ai_info.AI_TYPE_TO_METRICS[ai_types[0]]:
            metric = f"{METRIC_CBTOOL_PREFIX}{m}"
            d = self.aggregatePerfForAiNameAndMetric(df, d, metric, ai_name)
        return toSingleRowDF(d)

    def aggregatePerf(self, df):
        self.split_interval = self.getSplitInterval(df)
        ts = self.split_interval
        df = df.loc[dfInterval(df, *ts), :]
        ai_names = df["ai_name"].unique()
        results = pd.DataFrame()
        for ai_name in ai_names:
            result = self.aggregatePerfForAiName(df, ai_name)
            results = results.append(result, ignore_index=True)
        return results

    def computeAINameToHostAndTypeMap(self, df):
        df = df.loc[df["expid"] == self.expid, :]
        ai_name_to_host_and_type = {}
        for _, row in df.iterrows():
            t = row["type"]
            host = IP_TO_HOST[row["host_name"]]
            ai_name = row["ai_name"]

            new_record = (host, t)
            present_record = ai_name_to_host_and_type.get(ai_name, new_record)
            if present_record != new_record:
                raise ValueError(f"{ai_name} - two vms give different results {present_record} vs {new_record}")
            ai_name_to_host_and_type[ai_name] = new_record
        return ai_name_to_host_and_type


class SchedulerExperimentSeries:
    def __init__(self, base_path, config, ai_count):
        self.type = "scheduler"
        self.base_path = base_path
        _, self.name = os.path.split(base_path)
        self.ai_role_count = ai_info.AI_ROLE_TO_COUNT.copy()
        self.rescale_map = config.rescale_map  # hostname to exp_series
        self.ai_count = ai_count
        self.ai_types = config.tasks

        self.experiments = dict()
        self.dfs = {}
        self.df = None
        self.schedules = pd.DataFrame()

        if config.ai_role_count:
            self.ai_role_count.update(config.ai_role_count)

        for exp_match in self.getExperimentPathsMatches(base_path):
            composition_id, shuffle_id, custom_scheduler = exp_match.groups()
            composition_id = int(composition_id)
            shuffle_id = int(shuffle_id)
            custom_scheduler = "" if custom_scheduler is None else str(custom_scheduler)
            exp = SchedulerExperimentRecord(base_path, exp_match.string,
                                            composition_id, shuffle_id, custom_scheduler, self.ai_types, self)
            self.experiments[(composition_id, shuffle_id, custom_scheduler)] = exp
        self.readPerf()
        self.aggregatePerf()
        self.rescalePerf()
        self.computeCost()

    def getPerfMetricsForType(self, t1):
        return METRIC_CBTOOL_PREFIX + ai_info.AI_TYPE_TO_METRICS[t1][0]

    def readPerf(self):
        print("Getting perf data")
        perf = pd.DataFrame()
        for k, exp in self.experiments.items():
            composition_id, shuffle_id, custom_scheduler = k
            df = readExp(exp)
            df["expid"] = exp.expid
            df["composition_id"] = composition_id
            df["shuffle_id"] = shuffle_id
            df["custom_scheduler"] = custom_scheduler
            perf = perf.append(df, ignore_index=True)
            # TODO count number of tasks per type
        self.dfs["perf"] = perf

    def aggregatePerf(self):
        results = pd.DataFrame()
        perf = self.dfs["perf"]
        for k, exp in self.experiments.items():
            df = perf.loc[(perf["expid"] == exp.expid), :]
            result = exp.aggregatePerf(df)
            results = results.append(result, ignore_index=True)
        self.dfs["perf_agg"] = results
        self.df = results

    def rescalePerf(self):
        for expid in self.df["expid"].unique():
            df = self.df.loc[self.df["expid"] == expid]
            for ai_name in df["ai_name"].unique():
                df2 = df.loc[df["ai_name"] == ai_name, :]
                host_name = df2["host_name"].min()
                t = df2["type"].min()
                for metric in ai_info.AI_TYPE_TO_METRICS[t]:
                    factor = self.rescale_map[host_name][t][metric]
                    select = (self.df["expid"] == expid) & (df["ai_name"] == ai_name)
                    for mt in ["avg_", "std_"]:
                        input_col = f"{mt}{metric}"
                        output_col = f"rescaled_{input_col}"
                        self.df.loc[select, output_col] = self.df.loc[select, input_col] / factor

    def computeCost(self):
        for expid in self.df["expid"].unique():
            df = self.df.loc[self.df["expid"] == expid]
            for ai_name in df["ai_name"].unique():
                df2 = df.loc[df["ai_name"] == ai_name, :]
                t = df2["type"].min()
                metric = ai_info.AI_TYPE_TO_METRICS[t][0]
                output_col = "cost"
                input_col = f"rescaled_avg_{metric}"
                select = (self.df["expid"] == expid) & (df["ai_name"] == ai_name)
                if metric == "throughput":
                    self.df.loc[select, output_col] = 1. / self.df.loc[select, input_col]
                else:
                    self.df.loc[select, output_col] = self.df.loc[select, input_col]
            select = self.df.expid == expid
            self.df.loc[select, "max_cost"] = self.df.loc[select, "cost"].max()

    def printExperimentResults(self):
        xs = []
        ys_map = {s: [] for s in sorted(self.df["scheduler"].unique())}
        for i, composition in enumerate(sorted(self.df["composition_id"].unique())):
            xs.append(i)
            for scheduler in sorted(self.df["scheduler"].unique()):
                select = (self.df["composition_id"] == composition) & (self.df["scheduler"] == scheduler)
                ys_map[scheduler].append(self.df.loc[select, "max_cost"].min())
        fig, ax = plt.subplots()
        for k, v in ys_map.items():
            ax.scatter(xs, v, label=k[1:])
        ax.legend()
        plt.show()
        return xs, ys_map

    @staticmethod
    def getExperimentPathsMatches(base_path):
        path_regex = "([0-9]{1,4})scheduler([0-9]{1,2})(_custom|_random|_round_robin|_default){0,1}"

        def matchExpidRegex(e):
            i = e.split("/")[-1]
            return re.fullmatch(path_regex, i)

        pattern = os.path.join(base_path, f"*")
        expids = glob.glob(pattern)
        matches = [matchExpidRegex(e) for e in expids]
        return [m for m in matches if bool(m)]

    def computeScheduleSummarySingle(self, exp, hosts, columns, index):
        result = pd.DataFrame(np.zeros((1, len(columns)), dtype=np.int32), index=index, columns=columns)
        ai_name_to_host_and_type = exp.computeAINameToHostAndTypeMap(self.df)

        for _, host_and_type in ai_name_to_host_and_type.items():
            result.loc[:, host_and_type] += 1
            result.loc[:, (host_and_type[0], "all")] += 1

        for host in hosts:
            for t in ("all",) + self.ai_types:
                result.loc[:, ("all", t)] += result.loc[:, (host, t)]
        return result

    # TODO shuffle id resilient
    def computeScheduleSummary(self):
        self.schedules = pd.DataFrame()
        hosts = list(self.df["host_name"].unique())
        hosts = sorted([IP_TO_HOST[h] for h in hosts])
        columns = pd.MultiIndex.from_product([["all"] + hosts, ("all",) + self.ai_types])
        for composition_id in sorted(self.df["composition_id"].unique()):
            for shuffle_id in [0]:
                for scheduler in sorted(self.df["scheduler"].unique()):
                    exp = self.experiments[(composition_id, shuffle_id, scheduler)]
                    index = pd.MultiIndex.from_tuples([(composition_id, scheduler)], names=["composition", "scheduler"])
                    result = self.computeScheduleSummarySingle(exp, hosts, columns, index)
                    self.schedules = self.schedules.append(result)

    def extractNodeToLoads(self, composition_id, shuffle_id, scheduler):
        results = {}
        schedule = self.schedules.loc[(composition_id, scheduler)]
        hosts = [h for h in schedule.index.levels[0] if h != "all"]
        for host in hosts:
            result = np.zeros(len(self.ai_types))
            for i, ai_type in enumerate(self.ai_types):
                result[i] = schedule[(host, ai_type)]
            results[host] = result
        return results

    def extractActualCosts(self, composition_id, shuffle_id, scheduler):
        xs = []
        values = []
        df = self.df
        select = (df["composition_id"] == composition_id) & (df["shuffle_id"] == shuffle_id) & (df["scheduler"] == scheduler)
        df = df.loc[select, :]
        for _, row in df.iterrows():
            values.append(row["cost"])
            host = IP_TO_HOST[row["host_name"]]
            t = row["type"]
            xs.append(f"{host} {t}")
        return xs, values


class SchedulerMeanMetricComputer:
    RecordId = collections.namedtuple("RecordId", "host type composition scheduler")

    def __init__(self, ai_types, node_to_coeffs, xs, ys_actual, ys_expected):
        self.ai_types = ai_types
        self.node_to_coeffs = node_to_coeffs
        self.xs = xs
        self.ys_actual = ys_actual
        self.ys_expected = ys_expected

    def computeMetricForType(self, t=None, metric_fn=mean_squared_error):
        data = self.getDataForType(t)
        _, ys_actual, ys_expected = zip(*data)
        return metric_fn(ys_actual, ys_expected)

    def getDataForType(self, t):
        data = zip(self.xs, self.ys_actual, self.ys_expected)
        if t is None:
            return data
        return [(x, y1, y2) for (x, y1, y2) in data if x.type == t]

    def computeMetrics(self, metric_fn=mean_squared_error):
        result = dict()
        result["all"] = self.computeMetricForType(metric_fn=metric_fn)
        for t in self.ai_types:
            result[t] = self.computeMetricForType(t, metric_fn)
        return result

    @staticmethod
    def createFromExpSeries(exp_series, node_to_coeffs):
        xs_result, ys_actual_result, ys_expected_result = [], [], []
        df = exp_series.df

        def toRecordId(x, c, s):
            host, t = x.split(" ")
            return SchedulerMeanMetricComputer.RecordId(host, t, composition, scheduler)

        for composition in df["composition_id"].unique():
            for scheduler in df["scheduler"].unique():
                node_to_loads = exp_series.extractNodeToLoads(composition, 0, scheduler)
                xs, ys_expected = computeExpectedCost(exp_series.ai_types, node_to_loads, node_to_coeffs)
                expected_cost_map = dict(zip(xs, ys_expected))

                xs, ys_actual = exp_series.extractActualCosts(composition, 0, scheduler)
                for x, y_actual in zip(xs, ys_actual):
                    xs_result.append(toRecordId(x, composition, scheduler))
                    ys_actual_result.append(y_actual)
                    ys_expected_result.append(expected_cost_map[x])
        return SchedulerMeanMetricComputer(exp_series.ai_types, node_to_coeffs, xs_result,
                                           ys_actual_result, ys_expected_result)


def computeExpectedCost(ai_types, node_to_loads, node_to_coefficients):
    values = []
    xs = []
    n = len(ai_types)
    for node in node_to_loads.keys():
        loads = node_to_loads[node]
        coeffs = node_to_coefficients[node]
        for i, ai_type in enumerate(ai_types):
            cost = 0.
            if loads[i] > 0:
                cost = 1.
                loads[i] -= 1.
                for j in range(n):
                    cost += loads[j] * coeffs[i][j]
                loads[i] += 1.
            values.append(cost)
            xs.append(f"{node} {ai_type}")
    return xs, values


def plotActualVsExpectedCost(exp_series, node_to_coefficients, composition_id):
    k = len(exp_series.df["scheduler"].unique())
    fig, axs = plt.subplots(1, k, figsize=(k * 5, 4))
    schedulers, actual_res, model_res = [], [], []

    def updateResultList(result, result_list, ymax):
        ymax = max([ymax] + result[1])
        result_list.append(result)
        return ymax

    ymax = 0
    for i, scheduler in enumerate(sorted(exp_series.df["scheduler"].unique())):
        schedulers.append(scheduler)
        actual = exp_series.extractActualCosts(composition_id, 0, scheduler)
        ymax = updateResultList(actual, actual_res, ymax)

        node_to_loads = exp_series.extractNodeToLoads(composition_id, 0, scheduler)
        model = computeExpectedCost(exp_series.ai_types, node_to_loads, node_to_coefficients)
        ymax = updateResultList(model, model_res, ymax)

    for i, (scheduler, actual, model) in enumerate(zip(schedulers, actual_res, model_res)):
        xs_model, values_model = model
        xs_actual, values_actual = actual
        ax = axs[i]

        ax.set_title(f"{scheduler[1:]} scheduler")
        ax.set_ylabel("Performance cost")
        ax.scatter(xs_model, values_model, label="predicted")
        ax.scatter(xs_actual, values_actual, label="observed")
        ax.tick_params('x', labelrotation=45)
        ax.set_ylim(ymin=0, ymax=ymax)
        ax.legend()
    plt.show()

