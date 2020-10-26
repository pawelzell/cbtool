AI_TYPE_TO_ROLE = {"redis_ycsb": ("ycsb", "redis"), "hadoop": ("hadoopmaster", "hadoopslave"),
                   "linpack": ("linpack",), "wrk": ("wrk", "apache"), "filebench": ("filebench",),
                   "sysbench": ("sysbench", "mysql"),
                   #"unixbench": ("unixbench",), "netperf": ("netclient", "netserver"),
                   #"memtier": ("memtier", "redis"),
                   "multichase": ("multichase",),
                   "oldisim": ("oldisimdriver", "oldisimlb", "oldisimroot", "oldisimleaf"),
                   "open_daytrader": ("client_open_daytrader", "geronimo", "mysql")}
                   #"mongo_ycsb": ("ycsb", "mongos", "mongo_cfg_server", "mongodb")}
AI_ROLE_TO_TYPE = {role: t for t, roles in AI_TYPE_TO_ROLE.items() for role in roles}

AI_TYPE_TO_METRICS = {t: ("latency", "throughput") for t, _ in AI_TYPE_TO_ROLE.items()}
AI_TYPE_TO_METRICS_OVERRIDE = {"linpack": ("throughput",), "filebench": ("throughput",),
                      "unixbench": ("throughput",), "netperf": ("bandwidth",),
                      "multichase": ("throughput", "completion_time", "quiescent_time")}
AI_TYPE_TO_METRICS.update(AI_TYPE_TO_METRICS_OVERRIDE)

AI_ROLE_TO_COUNT = {role: 1 for role in AI_ROLE_TO_TYPE.keys()}
AI_ROLE_TO_COUNT_OVERRIDE = {"hadoopslave": 2, "mongodb": 3, "oldisimleaf": 2}
AI_ROLE_TO_COUNT.update(AI_ROLE_TO_COUNT_OVERRIDE)

