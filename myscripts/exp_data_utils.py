import ai_info


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
