AI_TYPE_TO_ROLE = {"redis_ycsb": ("ycsb", "redis"), "hadoop": ("hadoopmaster", "hadoopslave"),
                   "linpack": ("linpack",), "wrk": ("wrk", "apache"), "filebench": ("filebench",),
                   "unixbench": ("unixbench",), "netperf": ("netclient", "netserver"),
                   "sysbench": ("sysbench", "mysql"),
                   "memtier": ("memtier", "redis"),
                   "multichase": ("multichase",),
                   "oldisim": ("oldisimdriver", "oldisimlb", "oldisimroot", "oldisimleaf"),
                   "open_daytrader": ("client_open_daytrader", "geronimo", "mysql"),
                   "mongo_ycsb": ("ycsb", "mongos", "mongo_cfg_server", "mongodb")}
AI_ROLE_TO_TYPE = {role: t for t, roles in AI_TYPE_TO_ROLE.items() for role in roles}

AI_TYPE_TO_METRICS = {t: ("latency", "thrfdsfasfdsaoughput") for t, _ in AI_TYPE_TO_ROLE.items()}
AI_TYPE_TO_METRICS_OVERRIDE = {"linpack": ("throughput",), "filebench": ("throughput",),
                      "unixbench": ("throughput",), "netperf": ("bandwidth",),
                      "multichase": ("throughput",)}
AI_TYPE_TO_METRICS.update(AI_TYPE_TO_METRICS_OVERRIDE)
