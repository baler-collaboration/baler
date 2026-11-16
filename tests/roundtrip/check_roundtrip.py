#!/usr/bin/env python
"""Characterization checks for a baler train -> compress -> decompress -> plot round trip.

Run from the repository root, after the four modes have run (see run_roundtrip.sh):

    python tests/roundtrip/check_roundtrip.py \
        --workspace CMS_workspace --project CMS_project_v1 \
        --metrics-out roundtrip_metrics.json [--timings timings.json]

Thresholds are read from tests/roundtrip/thresholds.json (override with --thresholds).
Exit status is 0 if every check passes, 1 otherwise. The metrics JSON is written
in both cases so failed runs keep their diagnostics.

Output file layouts this script relies on (from baler at ec4b178):
  - output/training/loss_data.npy: written by modules/training.py as
    np.array([train_loss, val_loss]) -> shape (2, n_epochs_run). Row 0 is the
    per-epoch training loss, row 1 the validation loss (when config.test_size == 0
    baler copies the training loss into row 1). n_epochs_run can be < config.epochs
    if early stopping triggers.
  - output/compressed_output/compressed.npz: keys data (n_rows, latent),
    names, normalization_features.
  - output/decompressed_output/decompressed.npz: keys data (n_rows, n_cols), names.
  - Latent size: baler.py sets latent_space_size = ceil(n_cols / compression_ratio)
    for 1D data (overriding any value in the config).
"""

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(workspace, project):
    path = os.path.join(
        "workspaces", workspace, project, "config", f"{project}_config.py"
    )
    spec = importlib.util.spec_from_file_location(f"{project}_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = types.SimpleNamespace()
    module.set_config(config)
    return config, path


def count_pdf_pages(path):
    """Rough page count from the raw PDF bytes (matplotlib writes uncompressed
    page dictionaries). Informational only; returns None if it cannot tell."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    n = len(re.findall(rb"/Type\s*/Page(?![s\w])", raw))
    return n or None


def markdown_summary(metrics):
    """Short markdown report: check results, key numbers, per-variable rms/std."""
    ok = metrics.get("all_passed")
    v = metrics.get("versions", {})
    lines = [
        f"## Round trip: {'✅ all checks passed' if ok else '❌ checks failed'}",
        "",
        f"Python {v.get('python')}, torch {v.get('torch')}, numpy {v.get('numpy')}; "
        f"thresholds: `{metrics.get('thresholds_status')}`",
        "",
    ]
    failed = [r for r in metrics.get("checks", []) if not r["passed"]]
    if failed:
        lines += ["**Failed checks:**", ""]
        lines += [f"- {r['check']}: {r['detail']}" for r in failed]
        lines.append("")
    t = metrics.get("timings_s", {})
    if t:
        lines += [
            "| " + " | ".join(t) + " |",
            "|" + "---|" * len(t),
            "| " + " | ".join(f"{x} s" for x in t.values()) + " |",
            "",
        ]
    if "final_train_loss" in metrics:
        lines.append(
            f"Final train loss: {metrics['final_train_loss']:.5g} "
            f"({metrics.get('epochs_run')}/{metrics.get('epochs_configured')} epochs)"
        )
    summ = metrics.get("rms_over_std_summary")
    if summ:
        lines += [
            f"rms(residual)/std(original): median {summ['median']:.4f}, "
            f"max {summ['max']:.4f} (`{summ['argmax'].split('.')[-1]}`)",
            "",
            "<details><summary>Per-variable rms/std</summary>",
            "",
        ]
        ref = metrics.get("reference")
        if ref:
            lines += [
                f"Reference (not gated): {ref['epochs']}-epoch run, median rms/std "
                f"{ref['median_rms_over_std']:.4f}, final loss {ref['final_train_loss']:.5g}",
                "",
                f"| variable | rms/std | limit | {ref['epochs']}-epoch ref |",
                "|---|---|---|---|",
            ]
        else:
            lines += ["| variable | rms/std | limit |", "|---|---|---|"]
        fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
        for name, pv in metrics["per_variable"].items():
            row = f"| {name.split('.')[-1]} | {fmt(pv['rms_over_std'])} | {pv['limit']:.4g} |"
            if ref:
                row += f" {fmt(pv.get('reference'))} |"
            lines.append(row)
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n\n"


class Checker:
    def __init__(self):
        self.results = []

    def check(self, name, ok, detail="", kind="structural"):
        """kind: "structural" (files, shapes, finiteness) or "threshold" (a
        limit from thresholds.json). Calibration may use runs that fail only
        threshold checks, never runs that fail structural ones."""
        self.results.append(
            {"check": name, "passed": bool(ok), "detail": detail, "kind": kind}
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else "")
        )
        return bool(ok)

    @property
    def all_passed(self):
        return all(r["passed"] for r in self.results)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workspace", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--metrics-out", required=True)
    p.add_argument("--thresholds", default=os.path.join(HERE, "thresholds.json"))
    p.add_argument("--timings", help="optional JSON of per-step wall times (seconds)")
    p.add_argument(
        "--summary-md",
        help="optional file to append a markdown summary to (e.g. $GITHUB_STEP_SUMMARY)",
    )
    args = p.parse_args(argv)

    with open(args.thresholds) as f:
        thresholds = json.load(f)
    max_ratio_default = float(thresholds["max_rms_over_std"])
    max_ratio_per_var = thresholds.get("per_variable_max_rms_over_std", {})
    max_final_loss = float(thresholds["max_final_train_loss"])
    reference = thresholds.get("_reference")  # informational only, never gated on

    config, config_path = load_config(args.workspace, args.project)
    out = os.path.join("workspaces", args.workspace, args.project, "output")
    paths = {
        "model": os.path.join(out, "compressed_output", "model.pt"),
        "compressed": os.path.join(out, "compressed_output", "compressed.npz"),
        "decompressed": os.path.join(out, "decompressed_output", "decompressed.npz"),
        "loss": os.path.join(out, "training", "loss_data.npy"),
        "original": config.input_path,
        "comparison_pdf": os.path.join(out, "plotting", "comparison.pdf"),
        "loss_pdf": os.path.join(out, "plotting", "Loss_plot.pdf"),
    }

    c = Checker()
    metrics = {
        "workspace": args.workspace,
        "project": args.project,
        "config": config_path,
        "thresholds_file": os.path.relpath(args.thresholds),
        "thresholds_status": thresholds.get("_status"),
        "reference": (
            {
                k: reference[k]
                for k in ("epochs", "median_rms_over_std", "final_train_loss")
            }
            if reference
            else None
        ),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    try:
        import torch

        metrics["versions"]["torch"] = torch.__version__
    except Exception:  # torch is not needed for the checks themselves
        pass

    def finish():
        metrics["checks"] = c.results
        metrics["all_passed"] = c.all_passed
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
        if args.summary_md:
            with open(args.summary_md, "a") as f:
                f.write(markdown_summary(metrics))
        n_fail = sum(not r["passed"] for r in c.results)
        print(
            f"\n{'ALL CHECKS PASSED' if c.all_passed else f'{n_fail} CHECK(S) FAILED'}"
            f" ({len(c.results)} total). Metrics: {args.metrics_out}"
        )
        return 0 if c.all_passed else 1

    if args.timings:
        try:
            with open(args.timings) as f:
                metrics["timings_s"] = json.load(f)
        except (OSError, ValueError) as e:  # informational only
            metrics["timings_s"] = None
            print(f"[INFO] could not read {args.timings}: {e}")

    def load_npz(key, keys):
        """Load an .npz output and the arrays it must contain. A missing key or
        an unreadable file is recorded as a failed check (returns None) rather
        than raised, so the metrics JSON and summary are still written."""
        path = paths[key]
        try:
            with np.load(path, allow_pickle=True) as f:
                arrays = {k: f[k] for k in keys}
        except Exception as e:
            c.check(f"readable: {path}", False, f"{type(e).__name__}: {e}")
            return None
        c.check(f"readable: {path}", True, "keys " + ", ".join(keys))
        return arrays

    def run_checks():
        # 1. Output files exist
        for key in ("model", "compressed", "decompressed"):
            c.check(f"exists: {paths[key]}", os.path.isfile(paths[key]))
        if not c.all_passed:
            return

        original = load_npz("original", ("data", "names"))
        if original is None:
            return
        orig_data, orig_names = original["data"], original["names"]
        n_rows, n_cols = orig_data.shape

        # 2. Latent shape, derived from the config the same way baler.py does
        expected_latent = math.ceil(n_cols / config.compression_ratio)
        metrics["expected_latent_size"] = expected_latent
        compressed = load_npz("compressed", ("data", "names", "normalization_features"))
        if compressed is not None:
            comp_shape = tuple(compressed["data"].shape)
            metrics["compressed_shape"] = list(comp_shape)
            c.check(
                "compressed shape",
                comp_shape == (n_rows, expected_latent),
                f"got {comp_shape}, expected {(n_rows, expected_latent)} "
                f"= (rows, ceil({n_cols}/{config.compression_ratio}))",
            )

        # 3. Decompressed data matches the original layout and is finite
        metrics["original_shape"] = list(orig_data.shape)
        decompressed = load_npz("decompressed", ("data", "names"))
        shape_ok = False
        if decompressed is not None:
            dec_data, dec_names = decompressed["data"], decompressed["names"]
            metrics["decompressed_shape"] = list(dec_data.shape)
            shape_ok = c.check(
                "decompressed shape == original shape",
                dec_data.shape == orig_data.shape,
                f"got {dec_data.shape}, expected {orig_data.shape}",
            )
            c.check(
                "decompressed names == original names",
                dec_names.shape == orig_names.shape
                and bool(np.all(dec_names == orig_names)),
            )
            dec_float = dec_data.astype(np.float64)
            n_nonfinite = int(
                np.size(dec_float) - np.count_nonzero(np.isfinite(dec_float))
            )
            c.check(
                "decompressed values finite",
                n_nonfinite == 0,
                f"{n_nonfinite} non-finite",
            )

        # 4. Training loss
        if c.check(f"exists: {paths['loss']}", os.path.isfile(paths["loss"])):
            try:
                loss = np.load(paths["loss"])
            except Exception as e:
                loss = None
                c.check(f"readable: {paths['loss']}", False, f"{type(e).__name__}: {e}")
            layout_ok = loss is not None and c.check(
                "loss_data layout is (2, n_epochs)",
                loss.ndim == 2 and loss.shape[0] == 2 and loss.shape[1] >= 1,
                f"shape {loss.shape}",
            )
            if loss is not None:
                metrics["loss_data_shape"] = list(loss.shape)
            if layout_ok:
                final_train = float(loss[0, -1])
                metrics["epochs_run"] = int(loss.shape[1])
                metrics["epochs_configured"] = int(config.epochs)
                metrics["final_train_loss"] = final_train
                metrics["final_val_loss"] = float(loss[1, -1])
                metrics["train_loss_per_epoch"] = [float(x) for x in loss[0]]
                c.check(
                    "final train loss finite",
                    math.isfinite(final_train),
                    f"{final_train:.6g}",
                )
                c.check(
                    "final train loss below threshold",
                    final_train < max_final_loss,
                    f"{final_train:.6g} < {max_final_loss}",
                    kind="threshold",
                )

        # 5. Per-variable reconstruction quality: rms(residual) / std(original)
        if shape_ok:
            orig_f = orig_data.astype(np.float64)
            residual = dec_float - orig_f
            rms = np.sqrt(np.mean(residual**2, axis=0))
            std = np.std(orig_f, axis=0)
            per_var = {}
            bad = []
            for name, r, s in zip(orig_names, rms, std):
                name = str(name)
                ratio = float(r / s) if s > 0 else None
                limit = float(max_ratio_per_var.get(name, max_ratio_default))
                per_var[name] = {
                    "rms_residual": float(r),
                    "std_original": float(s),
                    "rms_over_std": ratio,
                    "limit": limit,
                }
                if reference:
                    per_var[name]["reference"] = reference["rms_over_std"].get(name)
                if ratio is None or not math.isfinite(ratio) or ratio > limit:
                    bad.append(f"{name}={ratio}")
            ratios = [
                v["rms_over_std"]
                for v in per_var.values()
                if v["rms_over_std"] is not None
            ]
            metrics["per_variable"] = per_var
            metrics["rms_over_std_summary"] = {
                "median": float(np.median(ratios)) if ratios else None,
                "max": float(np.max(ratios)) if ratios else None,
                "argmax": max(per_var, key=lambda k: per_var[k]["rms_over_std"] or -1),
            }
            c.check(
                "per-variable rms/std within thresholds",
                not bad,
                "all within limits" if not bad else "exceeded: " + ", ".join(bad),
                kind="threshold",
            )

        # 6. Plots
        for key in ("comparison_pdf", "loss_pdf"):
            path = paths[key]
            size = os.path.getsize(path) if os.path.isfile(path) else 0
            c.check(f"non-empty: {path}", size > 0, f"{size} bytes")
        pages = count_pdf_pages(paths["comparison_pdf"])
        metrics["comparison_pdf_pages"] = pages
        print(f"[INFO] comparison.pdf page count (heuristic): {pages}")

    # Anything unforeseen (e.g. an artifact with an unexpected layout) becomes a
    # failed structural check, so a broken run still leaves its diagnostics.
    try:
        run_checks()
    except Exception as e:
        c.check("checks ran to completion", False, f"{type(e).__name__}: {e}")
    return finish()


if __name__ == "__main__":
    sys.exit(main())
