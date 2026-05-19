"""
check_additivity.py — Validate the additive cost model hypothesis.

For each N-layer design in the archive, checks whether:
  metric(N-layer) ≈ sum of metric(1-layer sub-blocks)

where each sub-block matches the per-layer config (in_size, out_size, weight_bw, RF, activation).

Usage:
  python dataset/check_additivity.py \
      --archive /global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult \
      --n-layers 2          # or 3
      --plot                # save scatter plots
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# JSON parsing helpers (mirrors catapult_asic_to_numpy.py conventions)
# ---------------------------------------------------------------------------

def _parse_bw(type_str):
    """ac_fixed<4,3,true> → 4, or None."""
    if not type_str:
        return None
    m = re.search(r'<\s*(\d+)', type_str)
    return int(m.group(1)) if m else None


def _parse_shape(shape_str):
    """'[4]' or '[None, 4]' → first integer, or None."""
    nums = re.findall(r'\d+', shape_str or '')
    return int(nums[0]) if nums else None


INCLUDED = {'Dense', 'relu', 'sigmoid', 'tanh', 'hard_sigmoid', 'hard_tanh', 'HardActivation', 'linear'}


def extract_dense_layers(report):
    """
    Return list of layer dicts, one per Dense layer:
      {in_size, out_size, weight_bw, rf, activation}
    Activation comes from the layer immediately following each Dense entry.
    Returns None if report is invalid.
    """
    qofr = report.get('QOFRSummary', {})
    if not qofr or qofr.get('total_area', 0) <= 0:
        return None

    cfg_model = report.get('Config', {}).get('HLSConfig', {}).get('Model', {})

    layer_summary = report.get('LayerSummary', [])
    filtered = [e for e in layer_summary if e.get('Layer Class', '') in INCLUDED]

    layers = []
    i = 0
    while i < len(filtered):
        entry = filtered[i]
        lclass = entry.get('Layer Class', '')

        if lclass != 'Dense':
            i += 1
            continue

        in_size  = _parse_shape(entry.get('Input Shape', ''))
        out_size = _parse_shape(entry.get('Output Shape', ''))
        weight_bw = _parse_bw(entry.get('Weight Type', ''))
        rf = int(entry.get('Reuse') or cfg_model.get('ReuseFactor', 1))

        # peek at the next entry for activation
        activation = 'linear'
        if i + 1 < len(filtered):
            next_class = filtered[i + 1].get('Layer Class', '')
            if next_class in {'relu', 'sigmoid', 'tanh', 'hard_sigmoid', 'hard_tanh', 'HardActivation', 'linear'}:
                activation = next_class
                i += 1  # consume the activation row

        if None not in (in_size, out_size, weight_bw):
            layers.append(dict(in_size=in_size, out_size=out_size,
                               weight_bw=weight_bw, rf=rf, activation=activation))
        i += 1

    return layers if layers else None


def parse_report(path):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    layers = extract_dense_layers(d)
    if not layers:
        return None
    qofr = d['QOFRSummary']
    return dict(
        layers=layers,
        area=float(qofr['total_area']),
        latency=float(qofr['latency_cycles']),
        thruput=float(qofr['thruput_cycles']),
    )


def layer_key(l):
    return (l['in_size'], l['out_size'], l['weight_bw'], l['rf'], l['activation'])


# ---------------------------------------------------------------------------
# Build 1-layer lookup
# ---------------------------------------------------------------------------

def _collect_json_paths(archive_dir, layer_tag, exclude):
    """
    Collect report JSON paths from mlp-<layer_tag> symlink dir only.
    Falls back to scanning top-level run_* dirs if the symlink dir is absent.
    """
    symlink = os.path.join(archive_dir, f'mlp-{layer_tag}')
    if os.path.isdir(symlink):
        run_dirs = sorted(glob.glob(os.path.join(symlink, 'run_*')))
    else:
        run_dirs = [
            d for d in sorted(glob.glob(os.path.join(archive_dir, 'run_*')))
            if not any(os.path.basename(d).startswith(ex) for ex in exclude)
        ]
    paths = []
    for run_dir in run_dirs:
        # resolve symlinks manually so glob works
        real_run = os.path.realpath(run_dir)
        paths.extend(sorted(glob.glob(os.path.join(real_run, 'reports', '*.json'))))
    return paths


def build_1layer_lookup(archive_dir, exclude):
    lookup = {}   # key → list of (area, latency, thruput)
    paths = _collect_json_paths(archive_dir, '1layer', exclude)
    total = 0
    for path in paths:
        r = parse_report(path)
        if r is None or len(r['layers']) != 1:
            continue
        total += 1
        k = layer_key(r['layers'][0])
        lookup.setdefault(k, []).append((r['area'], r['latency'], r['thruput']))

    print(f"1-layer lookup: {total} reports → {len(lookup)} unique configs")
    avg_lookup = {k: tuple(np.mean(v, axis=0)) for k, v in lookup.items()}
    return avg_lookup


# ---------------------------------------------------------------------------
# Additivity check for N-layer designs
# ---------------------------------------------------------------------------

def check_additivity(archive_dir, n_layers, exclude, lookup_1layer):
    paths = _collect_json_paths(archive_dir, f'{n_layers}layer', exclude)

    results = []       # (actual_area, pred_area, actual_lat, pred_lat, actual_thru, pred_thru)
    thru_theory = []   # (actual_thru, max_RF_theory)
    missing = 0

    for path in paths:
            r = parse_report(path)
            if r is None or len(r['layers']) != n_layers:
                continue

            pred_area = pred_lat = pred_thru = 0.0
            ok = True
            for lyr in r['layers']:
                k = layer_key(lyr)
                if k not in lookup_1layer:
                    missing += 1
                    ok = False
                    break
                a, l, t = lookup_1layer[k]
                pred_area  += a
                pred_lat   += l
                pred_thru  += t

            if ok:
                results.append((r['area'], pred_area, r['latency'], pred_lat, r['thruput'], pred_thru))

            # Throughput theory check: max(RF_i) regardless of 1-layer lookup
            if r is not None and len(r['layers']) == n_layers:
                max_rf = max(lyr['rf'] for lyr in r['layers'])
                thru_theory.append((r['thruput'], float(max_rf)))

    return results, missing, thru_theory


def report_thruput_theory(thru_theory, n_layers):
    actual = np.array([t[0] for t in thru_theory])
    pred   = np.array([t[1] for t in thru_theory])

    exact_match = np.sum(actual == pred)
    pct_exact   = 100.0 * exact_match / len(actual)
    err_pct     = 100 * np.abs(pred - actual) / np.where(actual > 0, actual, np.nan)

    ss_res = np.nansum((actual - pred) ** 2)
    ss_tot = np.nansum((actual - np.nanmean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    print(f"\n=== Throughput theory check: thruput = max(RF_i) ({n_layers}-layer) ===")
    print(f"  n designs     = {len(actual)}")
    print(f"  Exact match   = {exact_match} / {len(actual)} ({pct_exact:.1f}%)")
    print(f"  R²            = {r2:.6f}")
    print(f"  Abs error p50 = {np.nanpercentile(err_pct, 50):.2f}%")
    print(f"  Abs error p90 = {np.nanpercentile(err_pct, 90):.2f}%")
    print(f"  Abs error max = {np.nanmax(err_pct):.2f}%")


# ---------------------------------------------------------------------------
# Statistics & reporting
# ---------------------------------------------------------------------------

def report_stats(results, n_layers, metric, actual_idx, pred_idx):
    actual = np.array([r[actual_idx] for r in results])
    pred   = np.array([r[pred_idx]   for r in results])

    ratios = actual / np.where(pred > 0, pred, np.nan)
    err_pct = 100 * (pred - actual) / np.where(actual > 0, actual, np.nan)

    ss_res = np.nansum((actual - pred) ** 2)
    ss_tot = np.nansum((actual - np.nanmean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    print(f"\n  {metric}:")
    print(f"    n matched   = {len(results)}")
    print(f"    R²          = {r2:.4f}")
    print(f"    ratio mean  = {np.nanmean(ratios):.4f}  (actual/predicted; 1.0 = perfect)")
    print(f"    ratio std   = {np.nanstd(ratios):.4f}")
    print(f"    error mean  = {np.nanmean(err_pct):+.1f}%  (predicted vs actual)")
    print(f"    error p90   = {np.nanpercentile(np.abs(err_pct), 90):.1f}%")
    return actual, pred


def make_plots(results, n_layers, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    os.makedirs(outdir, exist_ok=True)
    metrics = [
        ('Area',       0, 1, 'area_sq'),
        ('Latency',    2, 3, 'latency_cycles'),
        ('Throughput', 4, 5, 'thruput_cycles'),
    ]
    for name, ai, pi, fname in metrics:
        actual = np.array([r[ai] for r in results])
        pred   = np.array([r[pi] for r in results])
        lim = max(actual.max(), pred.max()) * 1.05
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(actual, pred, s=8, alpha=0.4)
        ax.plot([0, lim], [0, lim], 'r--', lw=1)
        ax.set_xlabel(f'Actual {name}')
        ax.set_ylabel(f'Predicted {name} (sum of 1-layer)')
        ax.set_title(f'{n_layers}-layer additivity: {name}')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        out = os.path.join(outdir, f'{n_layers}layer_{fname}.png')
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Build 1-layer throughput lookup for inference-time derivation
# ---------------------------------------------------------------------------

def build_thruput_lookup(archive_dir, exclude):
    """
    Return a dict keyed by (in_size, out_size, weight_bw, rf) -> thruput_cycles.
    Built from all 1-layer designs in the archive. Activation type is excluded
    from the key because throughput is driven by the Dense computation (N*M/RF),
    not by the activation function. When multiple entries share the same key, the
    most common value is used.
    """
    from collections import Counter
    counts = defaultdict(Counter)
    paths = _collect_json_paths(archive_dir, '1layer', exclude)
    total = 0
    for path in paths:
        r = parse_report(path)
        if r is None or len(r['layers']) != 1:
            continue
        lyr = r['layers'][0]
        k = (lyr['in_size'], lyr['out_size'], lyr['weight_bw'], lyr['rf'])
        counts[k][r['thruput']] += 1
        total += 1
    lookup = {k: ctr.most_common(1)[0][0] for k, ctr in counts.items()}
    print(f"Throughput lookup: {total} 1-layer reports -> {len(lookup)} unique (in,out,bw,rf) configs")
    return lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze layer composition in archive")
    parser.add_argument('--archive', required=True, help='Path to wa-hls4ml-catapult archive')
    parser.add_argument('--n-layers', type=int, default=2, choices=[2, 3],
                        help='Layer depth to validate (default: 2)')
    parser.add_argument('--exclude', nargs='*', default=['run_20260504', 'run_20260505',
                                                          'run_20260506', 'run_20260507_143427'],
                        help='Run name prefixes to exclude')
    parser.add_argument('--plot', action='store_true', help='Save scatter plots')
    parser.add_argument('--plot-dir', default='additivity_plots',
                        help='Output directory for plots (default: additivity_plots)')
    parser.add_argument('--build-lookup', action='store_true',
                        help='Build and save a 1-layer throughput lookup table')
    parser.add_argument('--lookup-out', default='thruput_lookup.pkl',
                        help='Output path for the throughput lookup pickle (default: thruput_lookup.pkl)')
    args = parser.parse_args()

    if args.build_lookup:
        import pickle
        print(f"Archive : {args.archive}")
        print(f"Exclude : {args.exclude}")
        print("\n--- Building throughput lookup ---")
        lookup = build_thruput_lookup(args.archive, args.exclude)
        with open(args.lookup_out, 'wb') as f:
            pickle.dump(lookup, f)
        print(f"Saved to: {args.lookup_out}")
        return

    print(f"Archive : {args.archive}")
    print(f"Checking: {args.n_layers}-layer additivity")
    print(f"Exclude : {args.exclude}")

    print("\n--- Building 1-layer lookup ---")
    lookup = build_1layer_lookup(args.archive, args.exclude)

    print(f"\n--- Checking {args.n_layers}-layer designs ---")
    results, missing, thru_theory = check_additivity(args.archive, args.n_layers, args.exclude, lookup)

    print(f"\nMatched pairs : {len(results)}")
    print(f"Unmatched (no 1-layer counterpart): {missing}")

    if not results:
        print("No matched pairs found — check archive paths or exclusion list.")
        sys.exit(1)

    print(f"\n=== Additivity statistics ({args.n_layers}-layer vs sum of 1-layer blocks) ===")
    report_stats(results, args.n_layers, 'Area',       0, 1)
    report_stats(results, args.n_layers, 'Latency',    2, 3)
    report_stats(results, args.n_layers, 'Throughput', 4, 5)

    report_thruput_theory(thru_theory, args.n_layers)

    if args.plot:
        print("\n--- Saving plots ---")
        make_plots(results, args.n_layers, args.plot_dir)


if __name__ == '__main__':
    main()
