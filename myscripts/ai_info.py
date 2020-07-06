AI_TYPE_TO_ROLE = {"redis_ycsb": ("ycsb", "redis"), "hadoop": ("hadoopmaster", "hadoopslave"),
                   "linpack": ("linpack",), "wrk": ("wrk", "apache"), "filebench": ("filebench",),
                   "unixbench": ("unixbench",), "netperf": ("netclient", "netserver"),
                   "sysbench": ("sysbench", "mysql")}
AI_TYPE_TO_METRICS = {"linpack": ("throughput",), "filebench": ("throughput",),
                      "unixbench": ("throughput",), "netperf": ("bandwidth",),
                      "hadoop": ("latency", "throughput"), "redis_ycsb": ("latency", "throughput"),
                      "wrk": ("latency", "throughput"), "sysbench": ("latency", "throughput")}
AI_ROLE_TO_TYPE = {role: t for t, roles in AI_TYPE_TO_ROLE.items() for role in roles}
