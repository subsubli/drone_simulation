"""Build shape-balanced, shareable subsets of soft/hard v2 into ONE folder.

Balance = equal #episodes per shape (whole episodes, seed-fixed selection).
Merged rows are interleaved round-robin across shapes, so even a front-prefix
of the file stays shape-balanced. episode_id is 0-indexed in emit order.
Outputs: <out>/<name>.csv.gz  +  <out>/MANIFEST.md (recipe + episode lists).
"""
import os, glob, gzip, csv, random

EX = os.path.expanduser("~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples")
OUT = os.path.expanduser("~/drone_simulation/balanced_subsets")
os.makedirs(OUT, exist_ok=True)

# (source shape_dataset dir, output basename, episodes-per-shape)
JOBS = [
    (f"{EX}/data_soft_v2/shape_dataset", "soft_v2_100k_balanced", 7),
    (f"{EX}/data_soft_v2/shape_dataset", "soft_v2_500k_balanced", 35),
    (f"{EX}/data_hard_v2/shape_dataset", "hard_v2_100k_balanced", 7),
    (f"{EX}/data_hard_v2/shape_dataset", "hard_v2_500k_balanced", 35),
]
SEED = 0

def shape_of(path):
    return os.path.basename(path).split("-")[0].split("_")[0]

manifest = ["# Balanced shareable subsets of soft/hard v2\n",
            "Balance = **equal episodes per shape** (whole episodes, seed-fixed).",
            "Rows are **interleaved round-robin** across shapes, so any front-prefix stays balanced.",
            f"Selection seed = {SEED}. Shapes = circle/pentagon/square/triangle (star is untrained, absent).\n",
            "| file | eps/shape | shapes | episodes | rows |", "|---|---|---|---|---|"]

for src, name, n in JOBS:
    files = sorted(glob.glob(os.path.join(src, "*.csv")))
    by_shape = {}
    for f in files:
        by_shape.setdefault(shape_of(f), []).append(f)
    shapes = sorted(by_shape)
    picked = {}
    for sh in shapes:
        lst = sorted(by_shape[sh])
        rng = random.Random(SEED)          # same seed per shape -> reproducible
        rng.shuffle(lst)
        picked[sh] = lst[:n]
    # round-robin interleave
    order = []
    for i in range(n):
        for sh in shapes:
            order.append(picked[sh][i])
    out_gz = os.path.join(OUT, name + ".csv.gz")
    with open(order[0], newline="") as f:
        header = next(csv.reader(f))
    total = 0
    with gzip.open(out_gz, "wt", newline="") as out_f:
        w = csv.writer(out_f)
        w.writerow(["episode_id"] + header)
        for eid, path in enumerate(order):
            with open(path, newline="") as in_f:
                r = csv.reader(in_f); next(r)
                for row in r:
                    w.writerow([eid] + row); total += 1
    # episode list file
    with open(os.path.join(OUT, name + ".episodes.txt"), "w") as lf:
        for eid, path in enumerate(order):
            lf.write(f"{eid}\t{os.path.basename(path)}\n")
    counts = " ".join(f"{sh}:{len(picked[sh])}" for sh in shapes)
    manifest.append(f"| {name}.csv.gz | {n} | {counts} | {len(order)} | {total} |")
    print(f"[built] {name}  eps/shape={n}  episodes={len(order)}  rows={total}  -> {out_gz}")

with open(os.path.join(OUT, "MANIFEST.md"), "w") as mf:
    mf.write("\n".join(manifest) + "\n")
print("[done] MANIFEST.md written to", OUT)
