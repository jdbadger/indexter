#!/usr/bin/env python3
"""Check the contents of `dist/` and smoke-test the built wheel.

Run after `uv build`. Used by `just build` and by both GitHub workflows
(design.md decision 5) so all three run exactly the same checks. Not part of
the `indexter` package: excluded from the sdist (`source-exclude`) and from
coverage.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

REQUIRED_WHEEL_PATHS = ("indexter/db/schema.sql", "indexter/skill/SKILL.md")
LICENSE_PATTERN = "*.dist-info/licenses/LICENSE"
EXCLUDED_SDIST_DIRS = ("eval", "scripts")

# Run inside the isolated smoke-test environment: creates a database through
# the installed package alone, exercising the one dependency most likely to
# break on a clean install -- the sqlite-vec loadable extension.
_DB_SMOKE_SCRIPT = """
import struct
import tempfile
from pathlib import Path

from indexter.db.connection import open_db

with tempfile.TemporaryDirectory() as d:
    db_path = Path(d) / "smoke.db"
    vector = struct.pack("384f", *([0.0] * 384))
    with open_db(db_path, repo=d) as conn:
        conn.execute(
            "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (?, ?, ?, ?)",
            (1, "function", "python", vector),
        )
        count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    assert count == 1, f"expected 1 row, got {count}"
print("indexter smoke: sqlite-vec OK")
"""


class CheckFailed(Exception):
    """A distribution check or smoke test failed; the message names why."""


def find_dist(dist_dir: Path) -> tuple[Path, Path]:
    """Locate exactly one wheel and one sdist in `dist_dir`."""
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))

    if len(wheels) != 1:
        raise CheckFailed(f"expected exactly one wheel in {dist_dir}, found {[w.name for w in wheels]}")
    if len(sdists) != 1:
        raise CheckFailed(f"expected exactly one sdist in {dist_dir}, found {[s.name for s in sdists]}")

    return wheels[0], sdists[0]


def check_wheel(wheel_path: Path) -> None:
    """The wheel must ship the schema, the skill and the license, and must
    ship no test code.
    """
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()

    missing = [path for path in REQUIRED_WHEEL_PATHS if path not in names]
    if not any(fnmatch(name, LICENSE_PATTERN) for name in names):
        missing.append(LICENSE_PATTERN)

    excluded = sorted(
        name for name in names if "tests" in PurePosixPath(name).parts or PurePosixPath(name).name == "conftest.py"
    )

    problems = []
    if missing:
        problems.append(f"missing required paths: {', '.join(missing)}")
    if excluded:
        problems.append(f"contains excluded paths: {', '.join(excluded)}")
    if problems:
        raise CheckFailed(f"{wheel_path.name}: " + "; ".join(problems))


def check_sdist(sdist_path: Path) -> None:
    """The sdist must ship no eval or dev-script code."""
    with tarfile.open(sdist_path) as archive:
        names = archive.getnames()

    excluded = sorted(name for name in names if set(PurePosixPath(name).parts[1:]) & set(EXCLUDED_SDIST_DIRS))
    if excluded:
        raise CheckFailed(f"{sdist_path.name}: contains excluded paths: {', '.join(excluded)}")


def _project_version(repo_root: Path) -> str:
    result = subprocess.run(
        ["uv", "version", "--short"],  # noqa: S603, S607 -- fixed argv, no shell, a trusted dev-only script
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _run_isolated(wheel_path: Path, *command: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 -- fixed argv, no shell, a trusted dev-only script
        [  # noqa: S607 -- resolved via PATH deliberately, like the rest of this script
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--managed-python",
            "--with",
            str(wheel_path),
            "--",
            *command,
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def smoke_test_version(wheel_path: Path, repo_root: Path) -> None:
    """Installing only the wheel gives an `indexter` command reporting the
    project's own version.
    """
    expected = f"indexter {_project_version(repo_root)}"
    result = _run_isolated(wheel_path, "indexter", "--version")
    if result.returncode != 0:
        raise CheckFailed(f"smoke test: `indexter --version` failed:\n{result.stderr}")

    actual = result.stdout.strip()
    if actual != expected:
        raise CheckFailed(f"smoke test: expected {expected!r}, got {actual!r}")


def smoke_test_database(wheel_path: Path) -> None:
    """Installing only the wheel can create a database, loading sqlite-vec."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "XDG_DATA_HOME": tmp, "XDG_CONFIG_HOME": tmp}
        result = _run_isolated(wheel_path, "python", "-c", _DB_SMOKE_SCRIPT, env=env)

    if result.returncode != 0:
        raise CheckFailed(f"smoke test: database creation failed:\n{result.stderr}")
    if "sqlite-vec OK" not in result.stdout:
        raise CheckFailed(f"smoke test: unexpected output:\n{result.stdout}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir", type=Path, default=Path("dist"), help="Directory holding the built distributions."
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Only check distribution contents; skip the isolated-install smoke test.",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent

    try:
        wheel_path, sdist_path = find_dist(args.dist_dir)
        print(f"Checking {wheel_path.name} ...")
        check_wheel(wheel_path)
        print(f"Checking {sdist_path.name} ...")
        check_sdist(sdist_path)
        if not args.skip_smoke:
            print("Smoke-testing the wheel in an isolated environment ...")
            smoke_test_version(wheel_path, repo_root)
            smoke_test_database(wheel_path)
    except CheckFailed as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
