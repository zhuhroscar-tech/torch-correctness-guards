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
}


def _load_guard(name: str):
    if name == "addcdiv-stale-scalar":
        from .guards import addcdiv_stale_scalar

        return addcdiv_stale_scalar
    if name == "as-strided-restride-oob":
        from .guards import as_strided_restride_oob

        return as_strided_restride_oob
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


def _print_report(guard_name: str, report, *, no_color: bool) -> None:
    if guard_name == "as-strided-restride-oob":
        _print_as_strided_report(report, no_color=no_color)
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
