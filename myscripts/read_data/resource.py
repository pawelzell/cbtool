import os
import glob
import pandas as pd
import ai_info
from read_data.utils import *


def getResDataPaths(expid, ai_no, role, resource):
    # Role in resources files has underscore replaced with hyphen
    formatted_role = role.replace("_", "-")
    paths = glob.glob(os.path.join(expid.path,
                                   f"resources/metric_pod_*{formatted_role}-ai-{ai_no+1}_{resource}.csv"))
    expected_count = expid.exp_series.ai_role_count[role]
    if len(paths) != expected_count:
        raise ValueError(f"Unexpected number of resources {len(paths)} != {expected_count} for " \
                         f"{expid.expid} {role} {ai_no+1} {resource}")
    return paths


def getResourceDatapointValues(d, ts, df):
    df = df.loc[dfInterval(df, *ts), ["datetime", "value"]]
    values = df["value"]
    if values.count() == 0:
        raise ValueError(f"No data samples for {d}")
    d.update({"avg_cpu": values.mean(), "std_cpu": values.std(), "cpu_samples_count": values.count()})
    return toSingleRowDF(d)


# DF: expid, t1, t2, ai_no, ai_role, tasks, avg_cpu, std_cpu, cpu_samples_count
def getResourceDatapointLinearExperiment(expid, ai_no, ai_role, tasks, ts, df):
    d = {"expid": expid.expid, "t1": expid.t1, "t2": expid.t2, "ai_no": ai_no + 1,
         "ai_role": ai_role, "tasks": tasks + 1}
    return getResourceDatapointValues(d, ts, df)


def getResourceDatapointSchedulerExperiment(expid, ai_no, ai_role, host, ts, df):
    d = {"expid": expid.expid, "composition_id": expid.composition_id, "shuffle_id": expid.shuffle_id,
         "scheduler": expid.custom_scheduler, "ai_no": ai_no+1, "ai_role": ai_role, "host": host}
    return getResourceDatapointValues(d, ts, df)


def getResourceDatapointsSingleFile(expid, ai_no, ai_role, tss, df):
    results = pd.DataFrame()
    if expid.exp_series.type == "linear":
        for tasks in range(ai_no, len(tss)):
            result = getResourceDatapointLinearExperiment(expid, ai_no, ai_role, tasks, tss[tasks], df)
            results = results.append(result, ignore_index=True)
    elif expid.exp_series.type == "scheduler":
        if len(tss) != 1:
            raise TypeError("Resource datapoints extraction for scheduler experiments "
                            f"expects 1 interval got {len(tss)}")
        ts = tss[0]
        # TODO compute ai map only once
        ai_name_to_host_and_role = expid.computeAINameToHostAndTypeMap(expid.exp_series.df)
        ai_name = formatAIName(ai_no+1)
        host, _ = ai_name_to_host_and_role[ai_name]
        result = getResourceDatapointSchedulerExperiment(expid, ai_no, ai_role, host, ts, df)
        results = results.append(result, ignore_index=True)
    else:
        raise TypeError(f"Unknown exp_series type {expid.exp_series.type}")
    return results


def getResourceDatapoints(expid, ai_no, ai_type, tss):
    results = pd.DataFrame()
    for ai_role in ai_info.AI_TYPE_TO_ROLE[ai_type]:
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
                    t = expid.t1 if ai_no == 0 else expid.t2
                    result = getResourceDatapoints(expid, ai_no, t, tss)
                    results = results.append(result, ignore_index=True)
    return results


def getCpuDataSchedulerExperiment(exp_series):
    print("Getting cpu data")
    results = pd.DataFrame()
    if exp_series.type != "scheduler":
        raise TypeError(f"Expected scheduler exp_series got {exp_series.type}")
    for _, exp in exp_series.experiments.items():
        ai_name_to_host_and_type = exp.computeAINameToHostAndTypeMap(exp_series.df)
        for ai_no in range(exp_series.ai_count):
            ai_name = formatAIName(ai_no+1)
            if ai_name not in ai_name_to_host_and_type.keys():
                print(f"WARNING: AI name missing from ai_name_to_host_and_type map: {ai_name}")
                continue
            _, ai_type = ai_name_to_host_and_type[ai_name]
            try:
                result = getResourceDatapoints(exp, ai_no, ai_type, [exp.split_interval])
                results = results.append(result, ignore_index=True)
            except ValueError:
                print(f"WARNING: No resource data for {exp.expid} {ai_no+1}")
    return results


def getCpuAggregate(cpu):
    print("Aggregating cpu data")
    cpu_sum = cpu.groupby(["expid", "t1", "t2", "tasks"], as_index=False).sum()
    cpu_sum.pop("ai_no")
    return cpu_sum


def getCpuAggregateSchedulerExperiment(cpu):
    print("Aggregating cpu data")
    cpu_sum = cpu.groupby(["expid", "composition_id", "shuffle_id", "scheduler", "host"], as_index=False).sum()
    cpu_sum.pop("ai_no")
    return cpu_sum


def getCpuDataAll(exp_series):
    cpu = getCpuData(exp_series)
    agg = getCpuAggregate(cpu)
    result = {"cpu": cpu, "cpu_agg": agg}
    exp_series.dfs.update(result)
    return result


def getCpuDataAllSchedulerExperiment(exp_series):
    cpu = getCpuDataSchedulerExperiment(exp_series)
    agg = getCpuAggregateSchedulerExperiment(cpu)
    return {"cpu": cpu, "cpu_agg": agg}
