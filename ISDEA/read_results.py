import os
import json
import pandas as pd


# \\:task = "WN18RR1"
task = "NELL9951"


print("DSSGNN")
total_summary = {"mrr": [], "hit@10": [], "hit@5": [], "hit@1": []}
for seed in (46, 45, 44, 43, 42):
    #
    results = {}
    for topk in (10, 5, 1):
        #
        path = os.path.join(
            "logs",
            "transform",
            # \\:"{:s}~dx2~0~dssgnn:_:e50-ss{:d}~l2-sm{:d}".format(task, seed, seed),
            "{:s}~dx2~1~dssgnn:_:e50-ss{:d}~l2-sm{:d}".format(task, seed, seed),
            "metrics.json",
        )
        with open(path, "r") as file:
            #
            data = json.load(file)
            results["hit@{:d}".format(topk)] = float(data["Hit@{:d}".format(topk)])
            results["mrr".format(topk)] = float(data["MRR"])
    results = {key: results[key] for key in sorted(results)}
    for key in results:
        #
        total_summary[key].extend(results[key])
    print(results)
df = pd.DataFrame(total_summary)

print("Neural LP")
for seed in (46, 45, 44, 43, 42):
    #
    results = {}
    for topk in (10, 5, 1):
        #
        path = os.path.join(
            "clone",
            "NeuralLP",
            "exps",
            # \\:"{:s}-ind-{:d}".format(task, seed),
            "{:s}-perm-ind-{:d}".format(task, seed),
            "top{:d}.txt".format(topk),
        )
        with open(path, "r") as file:
            #
            for line in file:
                #
                if line[:7] == "Hits at":
                    #
                    results["hit@{:d}".format(topk)] = float(line.strip().split()[-1])
                if line[:7] == "Mean Re":
                    #
                    results["mrr".format(topk)] = float(line.strip().split()[-1])
    results = {key: results[key] for key in sorted(results)}
    print(results)


print("DRUM")
for seed in (46, 45, 44, 43, 42):
    #
    results = {}
    for topk in (10, 5, 1):
        #
        path = os.path.join(
            "clone",
            "DRUM",
            "exps",
            # \\:"{:s}-ind-{:d}".format(task, seed),
            "{:s}-perm-ind-{:d}".format(task, seed),
            "top{:d}.txt".format(topk),
        )
        with open(path, "r") as file:
            #
            for line in file:
                #
                if line[:7] == "Hits at":
                    #
                    results["hit@{:d}".format(topk)] = float(line.strip().split()[-1])
                if line[:7] == "Mean Re":
                    #
                    results["mrr".format(topk)] = float(line.strip().split()[-1])
    results = {key: results[key] for key in sorted(results)}
    print(results)


print("GraIL")
for seed in (46, 45, 44, 43, 42):
    #
    results = {}
    for topk in (10, 5, 1):
        #
        path = os.path.join(
            "clone",
            "GraIL",
            "experiments",
            # \\:"{:s}-ind-{:d}".format(task, seed),
            "{:s}-perm-ind-{:d}".format(task, seed),
            "log_rank_test.txt".format(topk),
        )
        with open(path, "r") as file:
            #
            for line in file:
                #
                lastline = line
        (names, values) = lastline.strip().split(":")
        loc = [name.strip() for name in names.split("|")].index("Hits@{:d}".format(topk))
        results["hit@{:d}".format(topk)] = float([value.strip() for value in values.split("|")][loc])
        loc = [name.strip() for name in names.split("|")].index("MRR")
        results["mrr"] = float([value.strip() for value in values.split("|")][loc])
    results = {key: results[key] for key in sorted(results)}
    print(results)


print("NBFNet")
for seed in (46, 45, 44, 43, 42):
    #
    results = {}
    for topk in (10, 5, 1):
        #
        path = os.path.join(
            "clone",
            "NBFNet",
            "experiments",
            "NBFNet",
            # \\:"Ind{:s}Ind".format(task),
            "Ind{:s}PermInd".format(task),
            str(seed),
            "log.txt",
        )
        with open(path, "r") as file:
            #
            lines = [line for line in file]
        for i in range(len(lines)):
            #
            if ">" in lines[i]:
                #
                loc = i
        for line in lines[loc + 2 :]:
            #
            if "hits@{:d}_".format(topk) in line:
                #
                results["hit@{:d}".format(topk)] = float(line.strip().split(" ")[-1])
            if "mrr:" in line:
                #
                results["mrr".format(topk)] = float(line.strip().split(" ")[-1])
    results = {key: results[key] for key in sorted(results)}
    print(results)
