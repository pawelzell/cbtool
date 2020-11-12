import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_config


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


def dfTypePair(df, t1, t2):
    return (df["t1"] == t1) & (df["t2"] == t2)


def getSplitIntervals(df, method="max", trace=None):
    if len(df["exp_id"].unique()) != 1:
        raise ValueError("getSplitInterval function expects df of a single experiment.")

    if trace is None:
        ais = df["ai_name"].unique()
        tss = [df.loc[df["ai_name"] == ai, "datetime"].min() for ai in ais]
        tss.append(df["datetime"].max())
        tss.sort()

        if method == "max":
            return list(zip(tss[:-1], tss[1:]))
        if method == "first_plus_interval":
            begins = tss[:-1]
            begins = [a + pd.Timedelta(minutes=global_config.INTERVAL_OFFSET_LEFT) for a in begins]
            interval = pd.Timedelta(minutes=20 - global_config.INTERVAL_OFFSET_LEFT -
                                            global_config.INTERVAL_OFFSET_RIGHT)
            #maxs = tss[1:]
            ends = [b + interval for b in begins]
            return list(zip(begins, ends))
        else:
            raise KeyError(f"Method not supported {method}")

    max_ai = len(df["ai_name"].unique())
    durations = trace.intervals
    if max_ai != len(durations):
        raise ValueError(f"Number of ai_name in df differs with number read from trace file {max_ai} vs "
                         f"{len(durations)}")

    def aiDatetimes(ai_no):
        return df.loc[df["ai_name"] == formatAIName(ai_no), "datetime"]

    def intervalMax(ai_no, duration):
        a = aiDatetimes(ai_no).min()
        if duration <= 0:
            b = a
        elif ai_no != max_ai:
            b = aiDatetimes(ai_no+1).min()
        else:
            b = df["datetime"].max()
        return a, b

    def intervalFirstPlus(ai_no, duration):
        a = aiDatetimes(ai_no).min()
        if duration <= global_config.INTERVAL_OFFSET_LEFT + global_config.INTERVAL_OFFSET_RIGHT:
            b = a + pd.Timedelta(minutes=global_config.INTERVAL_OFFSET_LEFT)
        else:
            b = a + pd.Timedelta(minutes=duration - global_config.INTERVAL_OFFSET_LEFT -
                                 global_config.INTERVAL_OFFSET_RIGHT)
        return a, b

    if method == "max":
        getInterval = intervalMax
    elif method == "first_plus_interval":
        getInterval = intervalFirstPlus
        #print(f"first_plus_interval {global_config.INTERVAL_OFFSET_LEFT} {global_config.INTERVAL_OFFSET_RIGHT}")
    else:
        raise KeyError(f"Method not supported {method}")
    return [getInterval(i+1, dur) for i, dur in enumerate(durations)]


def splitByIntervals(df, tss):
    return [df[dfInterval(df, *ts)] for ts in tss]


def toSingleRowDF(d):
    d2 = {}
    for k, v in d.items():
        d2.update({k: pd.Series(v, index=[0])})
    return pd.DataFrame(d2)


def formatQuantileColumnName(m, q):
    return f"{m}_quantile{int(q * 100)}"


# ai_no starting from 1
def formatAIName(ai_no):
    return f"ai_{ai_no}"
