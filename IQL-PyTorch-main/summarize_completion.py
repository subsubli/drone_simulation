"""Progress completion: final signed net laps >= 2/3 of the PD mean."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def summarize(rows, threshold):
    if not rows or not all(math.isfinite(r['net_laps']) for r in rows):
        raise ValueError('Nonempty finite net-laps records required')
    completed = [r for r in rows if r['net_laps'] >= threshold]
    return dict(episodes=len(rows), completed=len(completed),
                completion_rate=len(completed)/len(rows),
                completed_and_survived=sum(not r['crashed'] for r in completed),
                completed_then_failed=sum(r['crashed'] for r in completed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', type=Path, default=Path('runs_x500/net_laps_20261115.json'))
    ap.add_argument('--inputs', nargs='+', type=Path, default=[
        Path('runs_x500/net_laps_20261115.json'),
        Path('runs_x650/ensemble128_20k/net_laps_best_20261115.json'),
        Path('runs_x650/dagger10_ensemble128_20k/net_laps_best_20261115.json')])
    ap.add_argument('--out', type=Path, default=Path('runs_x500/completion_20261115.json'))
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    pd = json.loads(args.baseline.read_text())['results']['PD']['episodes']
    pd_mean = mean(r['net_laps'] for r in pd)
    threshold = pd_mean*2/3
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('PD reference must have positive finite progress')
    report = dict(definition='Final signed net_laps >= (2/3)*mean PD net_laps over all reference episodes; no additional survival requirement',
                  baseline=str(args.baseline), baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  pd_episodes=len(pd), pd_mean_net_laps=pd_mean, threshold=threshold,
                  scope='One pooled fixed threshold for both shapes and all listed conditions; not per-seed/per-shape normalization',
                  sources={})
    reference = {(r['seed'], r['shape']): r['target_laps_budget'] for r in pd}
    for path in args.inputs:
        source = json.loads(path.read_text())
        results = {}
        for name, value in source['results'].items():
            rows = value['episodes']
            if {(r['seed'],r['shape']):r['target_laps_budget'] for r in rows} != reference:
                raise ValueError(f'Incompatible evaluation seeds/paths/budgets: {path}/{name}')
            results[name] = dict(summary=summarize(rows,threshold),
                by_shape={shape:summarize([r for r in rows if r['shape']==shape],threshold)
                          for shape in sorted({r['shape'] for r in rows})},
                episodes=[dict(seed=r['seed'],shape=r['shape'],net_laps=r['net_laps'],
                               completed=r['net_laps']>=threshold,crashed=r['crashed']) for r in rows])
            print(path.parent.name, name, results[name]['summary'])
        report['sources'][str(path)] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),results=results)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print('PD mean', pd_mean, 'threshold', threshold)


if __name__ == '__main__':
    main()
