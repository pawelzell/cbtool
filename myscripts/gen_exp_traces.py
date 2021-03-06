import argparse
import random
import sys
import os
import pandas as pd
types_default = ["redis_ycsb", "wrk", "hadoop", "linpack"]
hadoop_slave_no = 2
max_hadoop_count = 16
scheduler_exp_shuffles_count = 1

resource_constraints = "typealter {type} {role}_{resource}_{constraint}={value}\n"
hadoop_sut = "typealter hadoop sut=hadoopmaster->{}_x_hadoopslave\n"

prefix = \
"""cldattach kub TESTKUB
expid {}
vmcattach all
vmclist

cldalter ai_defaults run_limit=100000000
typealter open_daytrader load_level=5
typealter open_daytrader load_duration=100
typealter sysbench load_level=1
typealter oldisim load_level=1
typealter hadoop load_level=1
typealter wrk load_level=1
typealter linpack load_level=2
typealter redis_ycsb operation_count=450000
"""

ai_type_to_role = {"redis_ycsb": ("ycsb", "redis"), "hadoop": ("hadoopmaster", "hadoopslave"),
                   "linpack": ("linpack",), "wrk": ("wrk", "apache"), "filebench": ("filebench",),
                   "unixbench": ("unixbench",), "netperf": ("netclient", "netserver"),
                   "sysbench": ("sysbench", "mysql")}

use_custom_scheduler = "typealter {} {}_custom_scheduler={}\n"

wait_cmd = "waitfor {}m\n"

instance = \
"""
aiattach {}
vmlist
waitfor {}m
"""

description_line = "# {} {} {}\n"

attach_instance = "aiattach {} {}\n"

wait_for_all_ai_arrival = "waituntil AI ARRIVED={} increasing 20 7200\n"

suffix = \
"""
monextract all
clddetach
exit
"""


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def is_mixed_tasks_list_valid(tasks):
    hadoop_count = sum([1 for task in tasks if task == "hadoop"])
    return hadoop_count <= max_hadoop_count


def get_resource_constraints(resource_data_csv):
    results = []
    df = pd.read_csv(resource_data_csv)
    for row in df.itertuples():
        if row.resource == "cpu":
            requests = int(row.avg)
            limits = int(2 * row.avg)
        elif row.resource == "memory":
            continue
            #if row.type != "hadoop":
            # Set memory constraints only for hadoop
            #requests = int(row.avg)
            #limits = int(2 * row.avg)
        else:
            print(f"Unknown resource {row.resource}")
            continue
        for constraint, value in [("requests", requests), ("limits", limits)]:
            result = {"type": row.type, "role": row.role, "resource": row.resource}
            result.update({"constraint": constraint, "value": f"{value}{row.unit}"})
            results.append(result)
    return results


def gen_exp(expid, filename, tasks, interval, constraints, exp_type, exp_summary,
            async_until, custom_scheduler=None):
    with open(filename, "w") as f:
        f.write(description_line.format(exp_type, expid, exp_summary))
        f.write(prefix.format(expid))
        f.write(hadoop_sut.format(hadoop_slave_no))
        f.write("\n")

        if custom_scheduler is not None:
            for t, roles in ai_type_to_role.items():
                for role in roles:
                    f.write(use_custom_scheduler.format(t, role, custom_scheduler))
        f.write("\n")

        for c in constraints:
            f.write(resource_constraints.format(**c))
        f.write("\n")

        async_until = min(int(async_until), len(tasks))
        for i, task in enumerate(tasks, start=1):
            if i <= async_until:
                f.write(attach_instance.format(task, "async"))
                f.write(wait_cmd.format(0))
            else:
                f.write(attach_instance.format(task, ""))
                f.write(wait_cmd.format(interval))
            if i == async_until:
                f.write(wait_for_all_ai_arrival.format(i))
                f.write(wait_cmd.format(interval))
        f.write("\n")
        f.write(suffix)


def gen_mixed_tasks_list(types, task_count, max_tries=100):
    for _ in range(max_tries):
        tasks = []
        if min(len(types), task_count) > 2:
            tasks += random.sample(types, 3)
        tasks += random.choices(types, k=task_count - len(tasks))
        if is_mixed_tasks_list_valid(tasks):
            return tasks
    raise ValueError(f"Tried to sample a tasks list for mixed experiment and failed {max_tries} times")


def gen_exp_mixed(types, no, task_count, interval, constraints):
    basepath="../traces"
    basename = expid = f"{no}mixed"
    filename = os.path.join(basepath, basename)
    tasks = gen_mixed_tasks_list(types, task_count)
    exp_summary = ",".join(tasks)
    print(f"will generate {filename}")
    gen_exp(expid, filename, tasks, interval, constraints, "mixed", exp_summary)


def gen_exp_linear(x, y, no, task_count, interval, constraints, async_until):
    basepath="../traces"
    if x == y:
        basename = f"{x}"
    else:
        basename = f"{x}_{y}"
    filename = os.path.join(basepath, basename)
    expid = f"{no}{basename}"
    exp_summary = f"{x},{y},{task_count}"
    tasks = [x] + [y] * (task_count-1)
    print(f"will generate {filename} {exp_summary}")
    gen_exp(expid, filename, tasks, interval, constraints, "linear", exp_summary, async_until)


def gen_exp_scheduler(types, no, task_count, interval, constraints, async_until):
    basepath = "../traces"
    tasks = gen_mixed_tasks_list(types, task_count)
    for i in range(scheduler_exp_shuffles_count):
        random.shuffle(tasks)
        for custom_scheduler, suffix in [(None, "_default"), ("type-aware-scheduler", "_custom")]:
            basename = expid = f"{no}scheduler{i}{suffix}"
            exp_summary = ",".join(tasks)
            filename = os.path.join(basepath, basename)
            print(f"will generate {filename}")
            gen_exp(expid, filename, tasks, interval, constraints, "scheduler",
                    exp_summary, async_until, custom_scheduler=custom_scheduler)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CBTOOL experiments traces. "\
        "Generated experiment descriptions are saved to ../traces.")
    parser.add_argument("no", type=int, help="Each experiment has to have unique id. " \
        "Generated experiment ids have form '<order number><exp_type>' for consecutive "
        "order numbers. This parameter specifiy the order number to start from.")
    parser.add_argument("task_count", metavar="K", type=int, help="Number of tasks deployed in a single experiment.")
    parser.add_argument("-t", dest="types_file", type=str, default=None, help="File with list of task types (each type in a separate line. "\
        "Every task type has to be a valid CBTOOL application instance type e.g. redis_ycsb, hadoop, linpack")
    parser.add_argument("-tl", dest="types_list", type=str, default=None, help="Comma separate list of task types (each type in a separate line. "\
        "Every task type has to be a valid CBTOOL application instance type e.g. redis_ycsb, hadoop, linpack")
    parser.add_argument("-r", dest="resources_file", type=str, default=None, help="Csv file with cpu and memory average usage for  "\
        "each task type and vm type. Used to set cpu and memory requests and limits for each pod.")
    parser.add_argument("-m", dest="mode", default="linear", choices=["linear", "mixed", "scheduler"], help="Type of " \
        "generated experiments (default linear). 'linear' - deploys one task of type x and (N-1) tasks of type y." \
        "`mixed` - deploys N tasks of random types, but at least from three different types if possible." \
        "`scheduler` - simmilar to mixed but generates multiple experiments with the same task composition but" \
        "with different order of tasks.")
    parser.add_argument("-i", metavar="I", dest="interval", type=int, default=20, help="Number of minutes" \
        " to wait between deployment of two consecutive tasks.")
    parser.add_argument("-n", metavar="N", dest="experiment_count", type=int, default=3, help="If mixed mode " \
        "is selected, generates N different experiments.")
    parser.add_argument("-hadoop", metavar="H", dest="max_hadoop_count", type=int, default=None, \
        help="Prevent generation of an experiment with more than H hadoop tasks in order to avoid OOM.")
    parser.add_argument("-async", metavar="A", dest="async_until", type=int, default=0, 
                        help="Attach tasks asyncronously until first A tasks start.")
    return parser.parse_args()


def read_task_types_from_file(types_file):
    result = []
    with open(types_file, "r") as f:
        for t in f:
            result.append(t.strip())
    return result


def main():
    args = parse_args()
    constraints = []
    if args.resources_file:
        constraints = get_resource_constraints(args.resources_file)
    no = args.no
    if args.types_list:
        types = args.types_list.split(",")
    elif args.types_file:
        types = read_task_types_from_file(args.types_file)
    else:
        types = types_default
    if args.max_hadoop_count:
        global max_hadoop_count
        max_hadoop_count = args.max_hadoop_count

    if args.mode == "linear":
        for x in types:
            for y in types:
                task_count = args.task_count if y != "hadoop" else min(args.task_count, max_hadoop_count)
                gen_exp_linear(x, y, no, task_count, args.interval, constraints, args.async_until)
                no += 1
    elif args.mode == "mixed":
        for _ in range(args.experiment_count):
            gen_exp_mixed(types, no, args.task_count, args.interval, constraints)
            no += 1
    elif args.mode == "scheduler":
        for _ in range(args.experiment_count):
            gen_exp_scheduler(types, no, args.task_count, args.interval, constraints, args.task_count)
            no += 1
    else:
        print(f"Unsupported mode {args.mode}")
        return 1
    print(f"Hadoop slave number set to {hadoop_slave_no}")
    print(f"Resources constraints inferred from file {args.resources_file}")

if __name__ == "__main__":
    main()
