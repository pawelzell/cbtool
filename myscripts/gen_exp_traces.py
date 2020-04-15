import sys
import os

prefix = \
"""cldattach kub TESTKUB
expid {}{}
vmcattach all
vmclist

cldalter ai_defaults run_limit=100000000
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


def gen_exp(x, y, nr, interval="20m"):
    basepath="../traces"
    if x == y:
        basename=f"{x}"
    else:
        basename=f"{x}_{y}"
    filename = os.path.join(basepath, basename)
    print(f"will generate {filename} {x} {y} {nr}")
    with open(filename, "w") as f:
        f.write(prefix.format(nr, basename))
        f.write(instance.format(x, interval))
        for _ in range(5):
            f.write(instance.format(y, interval))
        f.write(suffix)

    return nr+1

def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <nr> <types>")
        return 1

    nr = int(sys.argv[1])
    for x in sys.argv[2:]:
        for y in sys.argv[2:]:
            nr = gen_exp(x, y, nr)

if __name__ == "__main__":
    main()
