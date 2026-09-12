"""January forecast and separate retrospective evaluation; see README.md."""
import argparse
from pathlib import Path
import sys
for folder in ('src', 'experiments'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))

from threadpoolctl import threadpool_limits

from triplex import run


def clear_generated_outputs(folder, clusters):
    """An explicit overwrite must not leave an old evaluator or manifest looking current."""
    names = ['run_manifest.json', 'independent_verification.json']
    for cluster in clusters:
        prefix = f'C{cluster}'
        names += [f'{prefix}_{suffix}' for suffix in (
            'rankings.csv', 'plot_plan.csv', 'calibration.csv', 'marker_model.npz',
            'report.json', 'retrospective_evaluation.csv', 'historical_sites.csv', 'id_aliases.csv')]
        names += [path.name for path in folder.glob(f'{prefix}_validation_[0-9][0-9][0-9][0-9].csv')]
    for name in names:
        path = folder / name
        if path.is_file():
            path.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', action='store_true', help='Generate a synthetic judge demonstration')
    p.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--clusters', type=int, nargs='+', choices=[1, 2], default=[1, 2])
    p.add_argument('--start-year', type=int, default=2001)
    p.add_argument('--target-year', type=int, default=2008)
    p.add_argument('--validation-years', type=int, nargs='+', default=[2005, 2006, 2007])
    p.add_argument('--alpha', type=float, default=30000.)
    p.add_argument('--plot-budget', type=int, help='Plots per cluster; default is 10%% of listed candidate plots')
    p.add_argument('--replicates', type=int, default=1, help='Plots per candidate-location, an explicit planning assumption')
    p.add_argument('--candidates', type=Path, help='Optional CSV: LINE_UNIQUE_ID, LOC, CROSS; no outcomes needed')
    p.add_argument('--predict-only', action='store_true', help='Skip target-season evaluation')
    p.add_argument('--overwrite', action='store_true', help='Allow replacing files in the specified output directory')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if not (0 < args.alpha < float('inf')) or args.replicates < 1 or args.threads < 1:
        p.error('alpha must be finite and positive; replicates and threads must be positive')
    if args.plot_budget is not None and args.plot_budget < 0:
        p.error('plot-budget must be nonnegative')
    if args.validation_years != sorted(set(args.validation_years)) or not (
        args.start_year < min(args.validation_years) <= max(args.validation_years) < args.target_year
    ):
        p.error('Use distinct increasing validation years strictly between start and target years')
    if args.sample and (args.start_year != 2001 or args.target_year != 2008 or args.candidates):
        p.error('Synthetic demo uses 2001-2008 and its own candidates')
    args.clusters = sorted(set(args.clusters))
    args.output_dir = args.output_dir or Path('outputs/judge_forecast' if args.sample else 'outputs/forecast_2008')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        p.error('Output directory is not empty; choose another directory or explicitly use --overwrite')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_generated_outputs(args.output_dir, args.clusters)
    # Large generated artifacts stay local even if the parent output folder is tracked.
    (args.output_dir / '.gitignore').write_text('*\n', encoding='utf-8')
    if args.sample:
        from demo_data import generate
        args.data_dir = args.output_dir / 'synthetic_data'
        generate(args.data_dir)
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == '__main__':
    main()
