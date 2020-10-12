import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def getSplitIntervals(df, method="max"):
    ais = df["ai_name"].unique()
    tss = [df.loc[df["ai_name"] == ai, "datetime"].min() for ai in ais]
    tss.append(df["datetime"].max())
    tss.sort()
    if method == "max":
        return list(zip(tss[:-1], tss[1:]))
    if method == "first_plus_interval":
        begins = tss[:-1]
        interval = pd.Timedelta(minutes=19)
        maxs = tss[1:]
        ends = [min(b + interval, m) for b, m in zip(begins, maxs)]
        return list(zip(begins, ends))
    else:
        raise KeyError(f"Method not supported {method}")


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
