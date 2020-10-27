import os
import glob
import pandas as pd
from read_data.utils import *


def getPerfAggregateForMetricHelper(df, d, metric, quantiles=(0.25, 0.5, 0.75)):
    perf = df[metric]
    m = metric[len("app_"):]
    d.update({f"avg_{m}": perf.mean(), f"std_{m}": perf.std()})
    d.update({f"{m}_samples_count": perf.count()})
    for q in quantiles:
        d.update({formatQuantileColumnName(m, q): perf.quantile(q)})
    return d


def getPerfAggregateForMetric(expid, df, ai_no, tasks, metric, d):
    df = df.loc[df[metric].notna(), :]
    if df.empty:
        msg = f"Performance aggregation failed: no datapoints for " \
              f"{expid.expid} ai_no={ai_no+1} tasks={tasks+1} metric={metric}"
        if ai_no == 0:  # We cannot analyze data without metrics for ai 0
            raise ValueError(msg)
        else:  # We do not care about "noise tasks" performance
            print(f"WARNING: {msg}")
            return d
    return getPerfAggregateForMetricHelper(df, d, metric)


def getPerfAggregateForAIAndTaskNumber(expid, df, ai_no, tasks, ts):
    d = {"expid": expid.expid, "t1": expid.t1, "t2": expid.t2, "ai_no": ai_no + 1, "tasks": tasks + 1}
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
            tss = getSplitIntervals(df, exp_series.getSplitIntervalMethod())
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


# Saves result in exp_series.dfs dictionary
def getPerfDataAll(exp_series):
    perf = getPerfData(exp_series)
    agg = getPerfAggregate(exp_series, perf)
    result = {"perf": perf, "perf_agg": agg}
    exp_series.dfs.update(result)
    return result
