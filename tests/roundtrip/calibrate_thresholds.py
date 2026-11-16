#!/usr/bin/env python
"""Derive tests/roundtrip/thresholds.json from several round-trip runs.

    python tests/roundtrip/calibrate_thresholds.py run1/roundtrip_metrics.json ... \
        --out tests/roundtrip/thresholds.json [--factor 1.5] [--source "<where the runs came from>"]

Rule: for each variable, limit = factor * max(rms/std over the runs); the final
training-loss ceiling is factor * max(final loss over the runs). The raw per-run
values are stored in the output so the thresholds can be audited and redone.
Prints the run-to-run spread (max/min per variable) and flags variables whose
spread exceeds 2x. With --reference, the metrics of a longer training run are
stored under "_reference" for context (shown in the CI summary, never gated on).
"""

import argparse
import json
import sys

# Checks that compare against a limit from thresholds.json. Metrics files
# written before checks carried a "kind" field are classified by name.
THRESHOLD_CHECKS = {
    "final train loss below threshold",
    "per-variable rms/std within thresholds",
}


def check_kind(result):
    return result.get(
        "kind", "threshold" if result["check"] in THRESHOLD_CHECKS else "structural"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("metrics", nargs="+", help="roundtrip_metrics.json files")
    p.add_argument("--out", required=True)
    # Defaults are the committed CI policy (see "_rule" in thresholds.json).
    p.add_argument("--factor", type=float, default=2.0)
    p.add_argument(
        "--floor",
        type=float,
        default=0.1,
        help="minimum per-variable rms/std limit. Rare-count variables (nearly all "
        "zeros, tiny std) make rms/std jump with a few miscounted rows.",
    )
    p.add_argument(
        "--include-threshold-failures",
        action="store_true",
        help="also use runs that failed only threshold checks (e.g. when "
        "recalibrating after a dependency update moved the metrics). Runs with "
        "structural failures (missing files, wrong shapes, non-finite values) are "
        "always rejected.",
    )
    p.add_argument("--source", default="", help="free text: where the runs came from")
    p.add_argument(
        "--reference",
        help="optional roundtrip_metrics.json of a longer training run, stored as an "
        "informational reference (not used for pass/fail)",
    )
    p.add_argument(
        "--reference-note", default="", help="free text describing the reference run"
    )
    args = p.parse_args(argv)

    runs, n_threshold_only = [], 0
    for path in args.metrics:
        with open(path) as f:
            m = json.load(f)
        failed = [r for r in m.get("checks", []) if not r["passed"]]
        structural = [r for r in failed if check_kind(r) != "threshold"]
        if structural or "per_variable" not in m or "final_train_loss" not in m:
            sys.exit(
                f"{path}: structural check(s) failed, not usable for calibration: "
                + ", ".join(r["check"] for r in structural)
            )
        if failed:
            if not args.include_threshold_failures:
                sys.exit(
                    f"{path}: run failed threshold check(s) "
                    f"({', '.join(r['check'] for r in failed)}). Pass "
                    "--include-threshold-failures to use it anyway (e.g. when "
                    "recalibrating after the metrics moved)."
                )
            n_threshold_only += 1
        runs.append(m)

    names = list(runs[0]["per_variable"])
    for path, m in zip(args.metrics, runs):
        if list(m["per_variable"]) != names:
            sys.exit(f"{path}: variable list differs from {args.metrics[0]}")

    per_var_raw = {
        n: [m["per_variable"][n]["rms_over_std"] for m in runs] for n in names
    }
    losses = [m["final_train_loss"] for m in runs]

    per_var_limit = {
        n: round(max(args.factor * max(v), args.floor), 6)
        for n, v in per_var_raw.items()
    }
    floored = [n for n, v in per_var_raw.items() if args.factor * max(v) < args.floor]
    loss_limit = round(args.factor * max(losses), 6)

    # max/min is undefined when a variable is reconstructed exactly (rms/std = 0)
    # in some run, e.g. integer variables after rounding; count those as wide.
    spread = {
        n: (max(v) / min(v) if min(v) > 0 else float("inf"))
        for n, v in per_var_raw.items()
    }
    wide = {n: s for n, s in spread.items() if s > 2}

    out = {
        "_status": "calibrated",
        "_rule": (
            f"per-variable max_rms_over_std = max({args.factor} x max over {len(runs)} runs, "
            f"floor {args.floor}); "
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
            # null where the minimum is 0 (spread undefined)
            "max_over_min_spread": {
                n: (round(s, 3) if s != float("inf") else None)
                for n, s in spread.items()
            },
        },
    }
    if args.reference:
        with open(args.reference) as f:
            ref = json.load(f)
        if list(ref["per_variable"]) != names:
            sys.exit(
                f"{args.reference}: variable list differs from the calibration runs"
            )
        out["_reference"] = {
            "_note": args.reference_note
            + " Informational only: not used for pass/fail.",
            "epochs": ref.get("epochs_run"),
            "versions": ref.get("versions"),
            "final_train_loss": ref["final_train_loss"],
            "median_rms_over_std": ref["rms_over_std_summary"]["median"],
            "rms_over_std": {n: ref["per_variable"][n]["rms_over_std"] for n in names},
        }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"{len(runs)} runs -> {args.out}")
    if n_threshold_only:
        print(f"{n_threshold_only} of them failed only threshold checks (included)")
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
    if floored:
        print(
            f"{len(floored)} limits raised to the floor {args.floor}: "
            + ", ".join(n.split(".")[-1] for n in floored)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
