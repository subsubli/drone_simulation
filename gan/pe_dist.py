"""Table-14-style |pos_err| distribution over a folder of per-episode CSVs (or a merged CSV).

    python pe_dist.py LABEL1=path/to/shape_dataset LABEL2=path/to/other/shape_dataset ...

Prints one markdown row per source: median, mean, p90, p99, max, off-path(>0.2m)% -- same columns
and 0.2m threshold as tables 7/13/14, so the class-conditional gen is directly comparable.
"""
import sys, os, csv, glob
import numpy as np

THR = 0.2  # off-path threshold, matches tables 7/13/14


def collect(path):
    files = ([path] if path.endswith('.csv') else glob.glob(os.path.join(path, '*.csv')))
    pes = []
    for fn in files:
        with open(fn) as f:
            rd = csv.DictReader(f)
            for r in rd:
                pes.append((float(r['tx-x']), float(r['ty-y']), float(r['tz-z'])))
    a = np.linalg.norm(np.array(pes), axis=1)
    return a


def main():
    print(f"| source | median | mean | p90 | p99 | max | off-path (>{THR}m) |")
    print("|---|---|---|---|---|---|---|")
    for arg in sys.argv[1:]:
        label, path = arg.split('=', 1)
        a = collect(path)
        print(f"| {label} | {np.median(a):.4f} | {a.mean():.3f} | {np.percentile(a,90):.3f} | "
              f"{np.percentile(a,99):.3f} | {a.max():.2f} | {(a>THR).mean()*100:.2f}% |")


if __name__ == '__main__':
    main()
