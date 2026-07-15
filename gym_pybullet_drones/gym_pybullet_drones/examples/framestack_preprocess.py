"""Frame-stack preprocessing for a merged shape_dataset CSV.

Turns each per-step state into a sliding window of the last 4 steps (the current step
plus the 3 previous ones), so the policy sees short-term history instead of a single
instant. Every STATE variable -- including the look-ahead lx/ly/lz -- is stacked; the
action (ax/ay/az), reward and done stay as the CURRENT step's values (they are the
prediction target / bookkeeping, not observation history).

Naming: each stacked variable gets a frame-index suffix appended directly to its name --
    <var>0   = current step        (t)
    <var>-1  = 1 step ago          (t-1)
    <var>-2  = 2 steps ago         (t-2)
    <var>-3  = 3 steps ago, oldest (t-3)
So e.g. lx -> lx0, lx-1, lx-2, lx-3 ; tx-x -> tx-x0, tx-x-1, tx-x-2, tx-x-3.

The window never crosses an episode boundary: at the first few steps of an episode
(where t-1/t-2/t-3 don't exist) the oldest available step is repeated (clamp), so every
row is fully populated and episode 0's history never leaks into episode 1.

Usage (NOT run automatically):
    python framestack_preprocess.py <in_merged.csv> <out_stacked.csv> [n_frames=4]
"""
import sys, csv
from itertools import groupby

#### The 16 state variables stacked over time (order preserved). Includes look-ahead.
STATE = ['tx-x', 'ty-y', 'tz-z', 'qx', 'qy', 'qz', 'qw',
         'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'lx', 'ly', 'lz']
#### Current-step-only columns (not stacked): action + bookkeeping.
CURRENT_ONLY = ['ax', 'ay', 'az', 'reward', 'done']


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python framestack_preprocess.py <in.csv> <out.csv> [n_frames=4]")
    inp, out = sys.argv[1], sys.argv[2]
    nf = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    rows = list(csv.reader(open(inp)))
    hdr = rows[0]
    ix = {n: i for i, n in enumerate(hdr)}
    data = rows[1:]
    if 'episode_id' not in ix:
        sys.exit("input CSV must have an episode_id column (use a merged.csv)")

    #### Build the new header: episode_id, step, then every state var x nf frames
    #### (suffix 0,-1,...,-(nf-1)), then the current-only columns.
    new_hdr = ['episode_id', 'step']
    for v in STATE:
        for k in range(nf):          # k = 0,1,2,3  ->  suffix 0,-1,-2,-3
            new_hdr.append(f"{v}{-k}")
    new_hdr += CURRENT_ONLY

    eid = [r[ix['episode_id']] for r in data]
    out_rows = []
    #### groupby keeps each episode's contiguous block (merged.csv is episode-ordered).
    pos = 0
    for _, grp in groupby(range(len(data)), key=lambda j: eid[j]):
        idxs = list(grp)             # global row indices of this one episode, in order
        for local, i in enumerate(idxs):
            row = [data[i][ix['episode_id']], data[i][ix['step']]]
            for v in STATE:
                col = ix[v]
                for k in range(nf):
                    #### clamp within the episode: t-k, but never before this episode's
                    #### first step (repeat the oldest available step instead of leaking
                    #### the previous episode or going out of range).
                    src = idxs[max(0, local - k)]
                    row.append(data[src][col])
            for c in CURRENT_ONLY:
                row.append(data[i][ix[c]])
            out_rows.append(row)
        pos += len(idxs)

    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(new_hdr)
        w.writerows(out_rows)

    n_state_cols = len(STATE) * nf
    print(f"[framestack] {inp} -> {out}")
    print(f"  rows={len(out_rows)}  episodes={len(set(eid))}  frames={nf}")
    print(f"  stacked state cols = {len(STATE)} vars x {nf} frames = {n_state_cols}")
    print(f"  total cols = {len(new_hdr)} (episode_id, step, {n_state_cols} state, {len(CURRENT_ONLY)} current-only)")
    print(f"  example suffixes for lx: " + ', '.join(f'lx{-k}' for k in range(nf)))


if __name__ == '__main__':
    main()
