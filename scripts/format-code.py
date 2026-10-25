"""Check handwritten source formatting; pass --write to apply it. Run from any directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(args, cwd=ROOT, **kwargs):
    executable = shutil.which(args[0])
    if not executable:
        raise SystemExit(f"Missing {args[0]}; see CONTRIBUTING.md for the development environment.")
    return subprocess.run([executable, *args[1:]], cwd=cwd, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="apply formatting (default: check only)")
    args = parser.parse_args()
    go_root = ROOT / "backend-go"
    go_files = [go_root / "main.go", *sorted((go_root / "internal").rglob("*.go"))]
    for folder in ("cmd",):
        go_files.extend(sorted((go_root / folder).rglob("*.go")))
    result = run(
        ["gofmt", "-w" if args.write else "-l", *map(str, go_files)],
        capture_output=True,
        text=True,
    )
    failed = bool(result.stdout.strip()) and not args.write
    if failed:
        print(result.stdout, end="")
    ruff_args = ["uvx", "--from", "ruff==0.14.1", "ruff", "format"]
    if not args.write:
        ruff_args.append("--check")
    run([*ruff_args, "app", "tests/test_compute_rpc.py"], cwd=ROOT / "ai_engine_python")
    run(["npm", "run", "format" if args.write else "format:check"], cwd=ROOT / "vue-grading-frontend")
    if failed:
        raise SystemExit("Go formatting differs; run this script with --write.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
