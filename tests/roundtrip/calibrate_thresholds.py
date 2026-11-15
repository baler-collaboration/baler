#!/usr/bin/env python
"""Derive tests/roundtrip/thresholds.json from several round-trip runs.

    python tests/roundtrip/calibrate_thresholds.py run1/roundtrip_metrics.json ... \
        --out tests/roundtrip/thresholds.json [--factor 1.5] [--source "<where the runs came from>"]

Rule: for each variable, limit = factor * max(rms/std over the runs); the final
training-loss ceiling is factor * max(final loss over the runs). The raw per-run
values are stored in the output so the thresholds can be audited and redone.
Prints the run-to-run spread (max/min per variable) and flags variables whose
spread exceeds 2x.
"""

import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("metrics", nargs="+", help="roundtrip_metrics.json files")
    p.add_argument("--out", required=True)
    p.add_argument("--factor", type=float, default=1.5)
    p.add_argument("--source", default="", help="free text: where the runs came from")
    args = p.parse_args(argv)

    runs = []
    for path in args.metrics:
        with open(path) as f:
            m = json.load(f)
        if not m.get("all_passed"):
            sys.exit(
                f"{path}: run did not pass its checks; not using it for calibration"
            )
        runs.append(m)

    names = list(runs[0]["per_variable"])
    for path, m in zip(args.metrics, runs):
        if list(m["per_variable"]) != names:
            sys.exit(f"{path}: variable list differs from {args.metrics[0]}")

    per_var_raw = {
        n: [m["per_variable"][n]["rms_over_std"] for m in runs] for n in names
    }
    losses = [m["final_train_loss"] for m in runs]

    per_var_limit = {n: round(args.factor * max(v), 6) for n, v in per_var_raw.items()}
    loss_limit = round(args.factor * max(losses), 6)

    spread = {n: max(v) / min(v) for n, v in per_var_raw.items()}
    wide = {n: s for n, s in spread.items() if s > 2}

    out = {
        "_status": "calibrated",
        "_rule": (
            f"per-variable max_rms_over_std = {args.factor} x max over {len(runs)} runs; "
            f"max_final_train_loss = {args.factor} x max over the same runs"
        ),
        "_source": args.source,
        "_environment": sorted(
            {
                f"python {m['versions'].get('python')}, torch {m['versions'].get('torch')}, "
                f"numpy {m['versions'].get('numpy')}"
                for m in runs
            }
        ),
        # Fallback for variables not listed below (none for the CMS example).
        "max_rms_over_std": 2.0,
        "per_variable_max_rms_over_std": per_var_limit,
        "max_final_train_loss": loss_limit,
        "_raw": {
            "final_train_loss": losses,
            "rms_over_std": per_var_raw,
            "max_over_min_spread": {n: round(s, 3) for n, s in spread.items()},
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"{len(runs)} runs -> {args.out}")
    print(
        f"final train loss: {min(losses):.5g}..{max(losses):.5g} -> ceiling {loss_limit:.5g}"
    )
    print(f"{'variable':28s} {'min':>7s} {'max':>7s} {'max/min':>7s} {'limit':>7s}")
    for n in names:
        v = per_var_raw[n]
        flag = "  <-- spread > 2x" if n in wide else ""
        print(
            f"{n.split('.')[-1]:28s} {min(v):7.4f} {max(v):7.4f} {spread[n]:7.2f} "
            f"{per_var_limit[n]:7.4f}{flag}"
        )
    if wide:
        print(f"\n{len(wide)} of {len(names)} variables have run-to-run spread > 2x.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
