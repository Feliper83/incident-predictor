"""Command-line entrypoint: generate synthetic metrics, build the
leading-indicator dataset, train on the earlier portion of the timeline,
evaluate on the later portion, and report both standard classification
metrics and the lead-time metric that actually matters for this problem.

    incident-predictor run
"""

from __future__ import annotations

import argparse
import sys

from .evaluation import evaluate, time_based_split
from .features import build_dataset
from .model import train_model
from .synthetic_data import generate_metrics_timeseries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="incident-predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Generate data, train, and evaluate end-to-end")
    run.add_argument("--hours", type=int, default=72, help="Hours of synthetic metrics to simulate")
    run.add_argument("--incidents", type=int, default=10)
    run.add_argument("--lookback", type=int, default=15, help="Minutes of history used for rolling features")
    run.add_argument("--lookahead", type=int, default=30, help="Predict incidents within this many minutes")
    run.add_argument("--train-fraction", type=float, default=0.7)
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--seed", type=int, default=42)

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    df = generate_metrics_timeseries(n_hours=args.hours, n_incidents=args.incidents, seed=args.seed)
    features, labels = build_dataset(df, lookback=args.lookback, lookahead=args.lookahead)

    X_train, y_train, X_test, y_test = time_based_split(features, labels, train_fraction=args.train_fraction)
    feature_columns = [c for c in features.columns if c != "minute"]

    print(
        f"Simulated {len(df)} minutes ({args.hours}h), {int(df['incident'].sum())} incidents. "
        f"{len(features)} labeled rows after feature warm-up -> {len(X_train)} train / {len(X_test)} test."
    )

    model = train_model(X_train[feature_columns], y_train, random_state=args.seed)
    report = evaluate(model, X_test, y_test, df, lookahead=args.lookahead, threshold=args.threshold)

    print(
        f"\nTest set: precision={report.precision:.2f} recall={report.recall:.2f} "
        f"f1={report.f1:.2f} roc_auc={report.roc_auc:.2f}"
    )
    print(
        f"Incidents in test window: {report.n_incidents_in_test}, "
        f"detected in advance: {report.n_incidents_detected}"
    )
    if report.average_lead_time_minutes is not None:
        print(f"Average lead time before incident: {report.average_lead_time_minutes:.1f} minutes")
    else:
        print("No incidents were detected in advance.")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
