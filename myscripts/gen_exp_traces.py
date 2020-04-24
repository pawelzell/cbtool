import sys
import os
import gen_exp_config as config

cpu_requests = "typealter {} {}_cpu_requests=1m\n"
cpu_limits = "typealter {} {}_cpu_limits={}m\n"

prefix = \
"""cldattach kub TESTKUB
expid {}{}
vmcattach all
vmclist

cldalter ai_defaults run_limit=100000000
typealter hadoop sut=hadoopmaster->hadoopslave
typealter hadoop load_level=1
typealter wrk load_level=1
typealter wrk load_duration=1
"""

instance = \
"""
aiattach {}
vmlist
waitfor {}
"""

suffix = \
"""
monextract all
clddetach
exit
"""


def gen_exp(x, y, nr, task_count, interval="20m"):
    basepath="../traces"
    if x == y:
        basename=f"{x}"
    else:
        basename=f"{x}_{y}"
    filename = os.path.join(basepath, basename)
    print(f"will generate {filename} {x} {y} {nr}")
    with open(filename, "w") as f:
        f.write(prefix.format(nr, basename))
        for ai_type, role, limit in config.cpu_limits:
            f.write(cpu_requests.format(ai_type, role))
            f.write(cpu_limits.format(ai_type, role, limit))
        f.write(instance.format(x, interval))
        for _ in range(task_count-1):
            f.write(instance.format(y, interval))
        f.write(suffix)

    return nr+1

def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <expid_nr> <max task count> <types>")
        return 1

    nr = int(sys.argv[1])
    task_count = int(sys.argv[2])
    for x in sys.argv[3:]:
        for y in sys.argv[3:]:
            nr = gen_exp(x, y, nr, task_count)

if __name__ == "__main__":
    main()
