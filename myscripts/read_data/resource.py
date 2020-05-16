import os
import glob
import pandas as pd
from read_data.utils import *


def getResDataPaths(expid, ai_no, role, resource):
    paths = glob.glob(os.path.join(expid.path, \
                                   f"resources/metric_pod_*{role}-ai-{ai_no+1}_{resource}.csv"))
    expected_count = expid.exp_series.ai_role_count[role]
    if len(paths) != expected_count:
        raise ValueError(f"Unexpected number of resources {len(paths)} != {expected_count} for " \
                         f"{expid.expid} {role} {ai_no+1} {resource}")
    return paths


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
                tss = getSplitIntervals(df, exp_series.getSplitIntervalMethod())
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
