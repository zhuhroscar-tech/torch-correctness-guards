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
}


def _load_guard(name: str):
    if name != "addcdiv-stale-scalar":  # defensive; argparse constrains this.
        raise KeyError(name)
    from .guards import addcdiv_stale_scalar

    return addcdiv_stale_scalar


def _print_report(report, *, no_color: bool) -> None:
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

        _print_report(report, no_color=args.no_color)
        return 0 if report["guard_fully_correct"] else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
