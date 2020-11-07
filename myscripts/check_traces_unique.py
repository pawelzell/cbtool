import sys
import os

TASKS_TO_ID = {"redis_ycsb": 0, "sysbench": 1, "hadoop": 2, "linpack": 3}


def countTasks(tasks):
    counts = [0.] * len(TASKS_TO_ID)
    for t in tasks:
        i = TASKS_TO_ID[t]
        counts[i] += 1
    return tuple(counts)


def getDuplicates(ids, signatures):
    s = set()
    duplicates = []
    for i, sig in zip(ids, signatures):
        if sig in s:
            duplicates.append(i)
        s.add(sig)
    return duplicates


def main():
    base_path = "../traces/"
    if len(sys.argv) != 3:
        print(f"Usage ./{sys.argv[0]} <c id start> <c id end>")
        return 0
    counts = []
    start, end = int(sys.argv[1]), int(sys.argv[2])+1
    ids = list(range(start, end))
    for i in ids:
        file_name = f"{i}scheduler0_custom"
        path = os.path.join(base_path, file_name)
        with open(path, "r") as f:
            line = f.readline()
            line = line.strip()
            tasks = line.split(" ")[3]
            tasks = tasks.split(",")
            counts.append(countTasks(tasks))

    duplicates = getDuplicates(ids, counts)
    if duplicates:
        print(f"Found duplicates: {duplicates}")
    for i, count in zip(ids, counts):
        print(f"{i}: {count}")

if __name__ == "__main__":
    main()
