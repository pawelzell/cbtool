import pandas as pd

from exp_data import *
from read_data.utils import *

OS_RESOURCES_PATH = "resources/metric_os.csv"
COLUMN_EXCLUDE_LIST = ("name", "time", "datetime")


# Returns dict with two dataframes - os_res, os_res_agg.
def readOsResources(exp_series, resource_columns=None):
    return readNodeResources(exp_series, OS_RESOURCES_PATH, "os_res")


def readCpuNodeResources(exp_series, node, resource_columns=None):
    result = readNodeResources(exp_series, f"resources/metric_node_{node}_cpu.csv", "cpu_node")

    cpu_agg = result["cpu_node_agg"]
    cpu_agg.loc[:, "avg_cpu"] = cpu_agg.loc[:, "avg_value"]
    cpu_agg.loc[:, "std_cpu"] = cpu_agg.loc[:, "std_value"]
    cpu_agg.pop("avg_value")
    cpu_agg.pop("std_value")
    return result


# Read node resources from single df
def readNodeResources(exp_series, resource_path, res_group_name, resource_columns=None):
    node_res_result = pd.DataFrame()
    node_res_agg_result = pd.DataFrame()
    for exp in exp_series.type_pair_to_exp.values():
        print(f"Read node resources for {exp.t1} {exp.t2} from {resource_path}")
        node_res, node_res_agg = readNodeResourcesExp(exp_series, exp, resource_path)
        node_res_result = node_res_result.append(node_res, ignore_index=True)
        node_res_agg_result = node_res_agg_result.append(node_res_agg, ignore_index=True)
    result = {f"{res_group_name}": node_res_result, f"{res_group_name}_agg": node_res_agg_result}
    exp_series.dfs.update(result)
    return result


def aggregateNodeResourcesForInterval(exp, df, tasks):
    d = {"expid": exp.expid, "t1": exp.t1, "t2": exp.t2, "tasks": tasks}
    for m in df.columns:
        if m not in COLUMN_EXCLUDE_LIST:
            d.update({f"avg_{m}": df[m].mean(), f"std_{m}": df[m].std()})
    return toSingleRowDF(d)


def getSplitIntervalsFromExp(exp_series, exp):
    perf = exp_series.dfs["perf"]
    perf = perf.loc[perf["expid"] == exp.expid, :]
    return getSplitIntervals(perf, exp_series.getSplitIntervalMethod())


def readNodeResourcesExp(exp_series, exp, resources_path):
    res_path = os.path.join(exp.path, resources_path)
    node_res = readResData(res_path)

    tss = getSplitIntervalsFromExp(exp_series, exp)
    node_res_split = splitByIntervals(node_res, tss)
    node_res_agg = pd.DataFrame()
    for i, node_res_agg_partial in enumerate(node_res_split):
        node_res_agg_partial = aggregateNodeResourcesForInterval(exp, node_res_agg_partial, i+1)
        node_res_agg = node_res_agg.append(node_res_agg_partial, ignore_index=True)
    return node_res, node_res_agg

# PerformanceResult
#   expid
#   df
#   agg_df
#   split_intervals
#   -> write some regression tests

# CpuResults
#  expid
#  df
#  agg_df
#  split_intervals
