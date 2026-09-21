"""Command-line interface for consolidated PyTorch correctness guards."""
from __future__ import annotations

import argparse
import json

from .style import print_fields, resolve_style, section, status_headline

_GUARDS = {
    "addcdiv-stale-scalar": {
        "description": "Inductor stale Python scalar in addcdiv_/addcmul_ Adam-style arithmetic",
        "module": "torch_correctness_guards.guards.addcdiv_stale_scalar",
    },
    "as-strided-restride-oob": {
        "description": "Inductor as_strided storage-span miscomputation for repeated+sliced restrided views",
        "module": "torch_correctness_guards.guards.as_strided_restride_oob",
    },
    "checkpoint-noise": {
        "description": "F.rrelu mutable noise buffer captured before fill under checkpoint/saved_tensors_hooks",
        "module": "torch_correctness_guards.guards.checkpoint_noise",
    },
    "dynamic-clamp": {
        "description": "Inductor stale automatically-dynamic Python float reused in torch.clamp bounds",
        "module": "torch_correctness_guards.guards.dynamic_clamp",
    },
    "normal-dtype-promotion": {
        "description": "torch.compile Normal.sample() silently promotes dtype away from eager loc dtype",
        "module": "torch_correctness_guards.guards.normal_dtype_promotion",
    },
    "shuffle-sample-frozen": {
        "description": "Dynamo freezes random.shuffle/random.sample results at trace time inside torch.compile",
        "module": "torch_correctness_guards.guards.shuffle_sample_frozen",
    },
}


def _load_guard(name: str):
    if name == "addcdiv-stale-scalar":
        from .guards import addcdiv_stale_scalar

        return addcdiv_stale_scalar
    if name == "as-strided-restride-oob":
        from .guards import as_strided_restride_oob

        return as_strided_restride_oob
    if name == "checkpoint-noise":
        from .guards import checkpoint_noise

        return checkpoint_noise
    if name == "dynamic-clamp":
        from .guards import dynamic_clamp

        return dynamic_clamp
    if name == "normal-dtype-promotion":
        from .guards import normal_dtype_promotion

        return normal_dtype_promotion
    if name == "shuffle-sample-frozen":
        from .guards import shuffle_sample_frozen

        return shuffle_sample_frozen
    raise KeyError(name)  # defensive; argparse constrains this.


def _print_addcdiv_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_native_diverges"]:
        print(status_headline(style, "fail", "torch.compile(inductor) stale-scalar addcdiv_/addcmul_ divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no stale-scalar divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "guard matches eager on every case, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "guard did NOT match eager on at least one case"))

    section("adam-step cases (iteration count -> eager vs compiled(native) vs compiled(guarded))")
    for c in report["cases"]:
        flag = "DIVERGES" if c["native_diverges"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"iters={c['iterations']}",
                    f"max|eager-native|={c['max_abs_diff_native']:.6e}  "
                    f"max|eager-guarded|={c['max_abs_diff_guarded']:.6e}  {flag:9s}  {guard_flag}",
                )
            ]
        )


def _print_as_strided_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_divergence_bug"]:
        print(status_headline(style, "warn", "as_strided restride divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no as_strided restride divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_as_strided() matches eager on every call, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "guard did NOT match eager on at least one call"))

    section("calls (index -> input values -> eager vs compiled vs guard)")
    for c in report["cases"]:
        flag = "DIVERGENCE" if c["divergence_bug"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        compiled_desc = c["compiled_error"] if not c["compiled_ok"] else str(c["compiled_value"])
        print_fields(
            [
                (
                    f"call {c['call_index']}",
                    f"x={c['x_values']!s:16s} eager={c['eager_value']}  "
                    f"compiled={compiled_desc}  {flag:10s}  {guard_flag}",
                )
            ]
        )


def _print_checkpoint_noise_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["bug_reproduced"]:
        print(status_headline(style, "warn", "F.rrelu checkpoint/saved-hooks gradient corruption reproduced on this host (#193671)"))
    else:
        print(status_headline(style, "info", "F.rrelu checkpoint/saved-hooks bug did NOT reproduce on this host's installed torch build"))

    if report["bug_reproduced"] and not report["mechanism_confirmed"]:
        print(status_headline(style, "warn", "bug reproduced but the expected root-cause signature did not fully match"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_rrelu() produces identical, NaN-free gradients across eager, checkpoint, and saved-tensors-hooks execution"))
    else:
        print(status_headline(style, "fail", "safe_rrelu() did NOT match expected behavior in at least one mode"))

    section("non-reentrant checkpoint")
    c = report["checkpoint_case"]
    print_fields(
        [
            ("buggy F.rrelu gradient diff vs uncheckpointed", f"{c['buggy_diff']!s}"),
            ("buggy forward-output diff (expected 0.0)", f"{c['buggy_forward_diff']!s}"),
            ("buggy diff with early-stop disabled (expected 0.0)", f"{c['no_early_stop_diff']!s}"),
            ("guard (safe_rrelu) gradient diff", f"{c['guard_diff']!s}"),
            ("guard has NaN", f"{c['guard_has_nan']!s}"),
        ]
    )

    section("saved_tensors_hooks (clone-based pack/unpack)")
    c = report["saved_hooks_case"]
    print_fields(
        [
            ("buggy F.rrelu gradient diff vs unhooked", f"{c['buggy_diff']!s}"),
            ("guard (safe_rrelu) gradient diff", f"{c['guard_diff']!s}"),
            ("guard has NaN", f"{c['guard_has_nan']!s}"),
        ]
    )


def _print_dynamic_clamp_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_stale_reuse_bug"]:
        print(status_headline(style, "warn", "stale dynamic-float clamp reuse reproduced on this host"))
    else:
        print(status_headline(style, "info", "no stale-reuse divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_clamp() matches eager on every call, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "guard did NOT match eager on at least one call"))

    section("calls (index -> shape/requires_grad/limit -> eager vs compiled vs guard)")
    for c in report["cases"]:
        flag = "STALE-REUSE" if c["stale_reuse_bug"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"call {c['call_index']}",
                    f"shape={c['shape']!s:10s} requires_grad={c['requires_grad']!s:5s} limit={c['limit']}  "
                    f"eager={c['eager_value']:.6f}  compiled={c['compiled_value']:.6f}  {flag:11s}  {guard_flag}",
                )
            ]
        )


def _print_normal_dtype_promotion_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_native_divergence"]:
        print(status_headline(style, "fail", "torch.compile Normal.sample() dtype-promotion divergence reproduced on this host (pytorch#194547)"))
    else:
        print(status_headline(style, "info", "no Normal.sample() dtype-promotion divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_compiled_normal_sample() restores eager's dtype-preservation contract on every case"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore eager's dtype contract on at least one case"))

    section("cases (loc dtype, scale dtype -> eager/compiled/guarded dtype)")
    for c in report["cases"]:
        native_flag = "DIVERGES" if c["dtype_diverges"] else "matches"
        guard_flag = "guard-ok" if c["guarded_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"loc={c['loc_dtype']:16s} scale={c['scale_dtype']:16s}",
                    f"eager={c['eager_dtype']:16s} compiled={c['compiled_dtype']:16s} "
                    f"native={native_flag:8s}  {guard_flag}",
                )
            ]
        )


def _print_shuffle_sample_frozen_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"]), ("seed", report["seed"]), ("calls", report["calls"])])

    if report["any_native_frozen"]:
        print(status_headline(style, "fail", "random.shuffle/random.sample frozen at trace time reproduced on this host (pytorch#197085)"))
    else:
        print(status_headline(style, "info", "no trace-time freeze reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_shuffle()/safe_sample() restore eager per-call randomness on every case"))
    else:
        print(status_headline(style, "fail", "guards did NOT restore eager semantics on at least one case"))

    section("cases (kind -> native vs guard vs eager)")
    for c in report["cases"]:
        native_flag = "FROZEN" if c["native_frozen_after_first_call"] else "ok"
        guard_flag = "guard-ok" if c["guard_correct"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["kind"],
                    f"native={native_flag:8s}  guard_matches_eager={str(c['guard_matches_eager']):5s}  {guard_flag}",
                )
            ]
        )


def _print_report(guard_name: str, report, *, no_color: bool) -> None:
    if guard_name == "as-strided-restride-oob":
        _print_as_strided_report(report, no_color=no_color)
    elif guard_name == "checkpoint-noise":
        _print_checkpoint_noise_report(report, no_color=no_color)
    elif guard_name == "dynamic-clamp":
        _print_dynamic_clamp_report(report, no_color=no_color)
    elif guard_name == "normal-dtype-promotion":
        _print_normal_dtype_promotion_report(report, no_color=no_color)
    elif guard_name == "shuffle-sample-frozen":
        _print_shuffle_sample_frozen_report(report, no_color=no_color)
    else:
        _print_addcdiv_report(report, no_color=no_color)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-guard",
        description="Run consolidated PyTorch correctness diagnostics and call-site guard checks.",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="list available guards")

    run_parser = subparsers.add_parser("run", help="run a guard diagnostic")
    run_parser.add_argument("guard", choices=sorted(_GUARDS))
    run_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    run_parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")

    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-correctness-guards {__version__}")
        return 0

    if args.command == "list":
        for name, meta in sorted(_GUARDS.items()):
            print(f"{name}\t{meta['description']}")
        return 0

    if args.command == "run":
        guard = _load_guard(args.guard)
        try:
            report = guard.diagnose()
        except guard.TorchUnavailableError as exc:
            if args.json:
                print(json.dumps({"guard": args.guard, "error": str(exc)}, indent=2))
            else:
                style = resolve_style(no_color_flag=args.no_color)
                print(status_headline(style, "fail", f"torch unavailable: {exc}"))
            return 2

        if args.json:
            payload = {"guard": args.guard, **report}
            print(json.dumps(payload, indent=2))
            return 0 if report["guard_fully_correct"] else 1

        _print_report(args.guard, report, no_color=args.no_color)
        return 0 if report["guard_fully_correct"] else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
