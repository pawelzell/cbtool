from read_data.os_resource import readOsResources
from sklearn import linear_model


NON_RESOURCES_COLUMN = ("datetime", "time", "t1", "t2", "expid", "tasks")


def getResourcesNamesSortedByCorrelation(exp_series, t1, t2):
    def isColumnEligible(df, column):
        return column not in NON_RESOURCES_COLUMN and \
            column.startswith("avg_") and \
            not df[column].isnull().values.any() and \
            df[df[column] <= 0.].empty

    def fitRegression(df, column):
        reg = linear_model.LinearRegression(normalize=True)
        df = df.loc[(df["t1"] == t1) & (df["t2"] == t2), :]
        xs = df["tasks"].to_numpy().reshape(-1, 1)
        ys = df[column].to_numpy().reshape(-1, 1)
        normalize_factor = ys[0][0]
        if normalize_factor != 0.:
            ys /= normalize_factor
        reg.fit(xs, ys)
        return reg.coef_[0], column

    # Read resources if not present
    if "os_res_agg" not in exp_series.dfs:
        readOsResources(exp_series)
    df = exp_series.dfs["os_res_agg"]
    columns = df.columns
    columns = [column for column in columns if isColumnEligible(df, column)]
    print("Fitting regressions")
    score_and_column = [fitRegression(df, column) for column in columns]
    return sorted(score_and_column, reverse=True)
