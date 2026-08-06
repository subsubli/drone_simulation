"""Build a real+diffusion MIXED dataset for the augmentation experiment.

Selects whole episodes (per-episode CSVs) from the real soft pool and the diffusion pool until
each hits its target row count, symlinks them into <out>/shape_dataset/ (so the later DAgger
re-merge can add episodes and re-index episode_id the standard way), then leaves the merge to
merge_shape_dataset.py. Real and diffusion per-episode CSVs share the exact same 22-column header.

    python build_mix.py --real-rows 1000000 --diff-rows 500000 --out mix_r1.0_d0.5
"""
import os, csv, glob, argparse


def episodes_upto(folder, target_rows):
    """Return sorted per-episode CSVs whose cumulative data-rows first reach target_rows."""
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    picked, total = [], 0
    for f in files:
        if total >= target_rows:
            break
        with open(f) as fh:
            n = sum(1 for _ in fh) - 1                 # minus header
        picked.append(f); total += n
    return picked, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-rows', type=int, required=True)
    ap.add_argument('--diff-rows', type=int, required=True)
    ap.add_argument('--real-dir', default='data_soft/shape_dataset')
    ap.add_argument('--diff-dir', default='../../../diffusion/gen_pool/shape_dataset')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()

    dst = os.path.join(A.out, 'shape_dataset')
    os.makedirs(dst, exist_ok=True)
    for old in glob.glob(os.path.join(dst, '*.csv')):
        os.unlink(old)                                 # clean re-runs

    real, rr = episodes_upto(A.real_dir, A.real_rows) if A.real_rows > 0 else ([], 0)
    diff, dr = episodes_upto(A.diff_dir, A.diff_rows) if A.diff_rows > 0 else ([], 0)
    for src in real + diff:
        link = os.path.join(dst, os.path.basename(src))
        os.symlink(os.path.abspath(src), link)

    print(f"[mix] real {len(real)} eps / {rr} rows  +  diff {len(diff)} eps / {dr} rows "
          f"= {rr + dr} rows -> {dst}")


if __name__ == '__main__':
    main()
