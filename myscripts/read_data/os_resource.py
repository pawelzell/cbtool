import pandas as pd

from exp_data import *
from read_data.utils import *

OS_RESOURCES_PATH = "resources/metric_os.csv"
COLUMN_EXCLUDE_LIST = ("name", "time", "datetime")


# Returns dict with two dataframes - os_res, os_res_agg.
def readOsResources(exp_series, resource_columns=None):
    os_res_result = pd.DataFrame()
    os_res_agg_result = pd.DataFrame()
    for exp in exp_series.type_pair_to_exp.values():
        os_res, os_res_agg = readOsResourcesExp(exp_series, exp)
        os_res_result = os_res_result.append(os_res, ignore_index=True)
        os_res_agg_result = os_res_agg_result.append(os_res_agg, ignore_index=True)
    result = {"os_res": os_res_result, "os_res_agg": os_res_agg_result}
    exp_series.dfs.update(result)
    return result

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


def aggregateOsResourcesForInterval(exp, df, tasks):
    d = {"expid": exp.expid, "t1": exp.t1, "t2": exp.t2, "tasks": tasks}
    for m in df.columns:
        if m not in COLUMN_EXCLUDE_LIST:
            d.update({f"avg_{m}": df[m].mean(), f"std_{m}": df[m].std()})
    return toSingleRowDF(d)


def getSplitIntervalsFromExp(exp_series, exp):
    perf = exp_series.dfs["perf"]
    perf = perf.loc[perf["expid"] == exp.expid, :]
    return getSplitIntervals(perf, exp_series.getSplitIntervalMethod())


def readOsResourcesExp(exp_series, exp):
    res_path = os.path.join(exp.path, OS_RESOURCES_PATH)
    os_res = readResData(res_path)

    tss = getSplitIntervalsFromExp(exp_series, exp)
    os_res_split = splitByIntervals(os_res, tss)
    os_res_agg = pd.DataFrame()
    for i, os_res_agg_partial in enumerate(os_res_split):
        os_res_agg_partial = aggregateOsResourcesForInterval(exp, os_res_agg_partial, i+1)
        os_res_agg = os_res_agg.append(os_res_agg_partial, ignore_index=True)
    return os_res, os_res_agg
