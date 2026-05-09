#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import math
import os
import re
import shlex
import subprocess
import sys
from typing import List, Tuple


@dataclasses.dataclass(frozen=True)
class TestCase:
    theta: float
    phi: float
    alpha: float
    exp_l1: float
    exp_l2: float
    exp_cost: float
    points: float
    tol: float


@dataclasses.dataclass(frozen=True)
class GraderConfig:
    executable: str
    build_with: str
    expect_output_count: int
    test_cases: List[TestCase]


_RE_KEYVAL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_RE_TEST_CASES_START = re.compile(r"^\s*TEST_CASES\s*=\s*\(\s*$")
_RE_TEST_CASE_LINE = re.compile(r'^\s*"(.+)"\s*$')


def _parse_float(s: str) -> float:
    return float(s.strip())


def _parse_test_case_row(row: str) -> TestCase:
    # Format (per grader.conf):
    # theta phi alpha | expected_l1 expected_l2 expected_cost | points | tol
    parts = [p.strip() for p in row.split("|")]
    if len(parts) != 4:
        raise ValueError(f"bad test case row (expected 4 '|' parts): {row!r}")

    inp = parts[0].split()
    exp = parts[1].split()
    pts = parts[2].split()
    tol = parts[3].split()

    if len(inp) != 3 or len(exp) != 3 or len(pts) != 1 or len(tol) != 1:
        raise ValueError(f"bad test case row (wrong field counts): {row!r}")

    return TestCase(
        theta=_parse_float(inp[0]),
        phi=_parse_float(inp[1]),
        alpha=_parse_float(inp[2]),
        exp_l1=_parse_float(exp[0]),
        exp_l2=_parse_float(exp[1]),
        exp_cost=_parse_float(exp[2]),
        points=_parse_float(pts[0]),
        tol=_parse_float(tol[0]),
    )


def parse_grader_conf(path: str) -> GraderConfig:
    executable = None
    build_with = None
    expect_output_count = None
    test_cases: List[TestCase] = []

    in_test_cases = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if in_test_cases:
                if line == ")":
                    in_test_cases = False
                    continue

                m = _RE_TEST_CASE_LINE.match(line)
                if not m:
                    raise ValueError(f"bad TEST_CASES line: {raw!r}")
                test_cases.append(_parse_test_case_row(m.group(1)))
                continue

            if _RE_TEST_CASES_START.match(line):
                in_test_cases = True
                continue

            m = _RE_KEYVAL.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if key == "EXECUTABLE":
                executable = val
            elif key == "BUILD_WITH":
                build_with = val
            elif key == "EXPECT_OUTPUT_COUNT":
                expect_output_count = int(val)

    if executable is None or build_with is None or expect_output_count is None:
        raise ValueError("grader.conf missing one of EXECUTABLE/BUILD_WITH/EXPECT_OUTPUT_COUNT")
    if not test_cases:
        raise ValueError("grader.conf contained no TEST_CASES")

    return GraderConfig(
        executable=executable,
        build_with=build_with,
        expect_output_count=expect_output_count,
        test_cases=test_cases,
    )


def _run(cmd: List[str], *, cwd: str) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _parse_solver_stdout(stdout: str, expect_n: int) -> List[float]:
    # Accept extra whitespace/newlines; use first nonempty line tokens.
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != expect_n:
            raise ValueError(f"expected {expect_n} outputs, got {len(toks)}: {line!r}")
        return [float(t) for t in toks]
    raise ValueError("no non-empty stdout lines")


def _max_abs_err(got: List[float], exp: List[float]) -> float:
    return max(abs(a - b) for a, b in zip(got, exp))


def _fmt_float(x: float) -> str:
    if math.isfinite(x):
        return f"{x:.10f}"
    return str(x)


def main(argv: List[str]) -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    grader_path = os.path.join(repo_root, "grader.conf")

    conf = parse_grader_conf(grader_path)

    # Build
    build_cmd = shlex.split(conf.build_with)
    rc, out, err = _run(build_cmd, cwd=repo_root)
    if rc != 0:
        sys.stderr.write("BUILD FAILED\n")
        sys.stderr.write(out)
        sys.stderr.write(err)
        return 2

    exe = conf.executable
    if exe.startswith("./"):
        exe_path = os.path.join(repo_root, exe[2:])
    else:
        exe_path = os.path.join(repo_root, exe)

    total_points = 0.0
    earned_points = 0.0
    worst_case = None  # (idx, maxerr)

    for i, tc in enumerate(conf.test_cases):
        total_points += tc.points
        cmd = [exe_path, str(tc.theta), str(tc.phi), str(tc.alpha)]
        rc, out, err = _run(cmd, cwd=repo_root)
        if rc != 0:
            sys.stdout.write(f"[{i}] FAIL (exit {rc}) theta={tc.theta} phi={tc.phi} alpha={tc.alpha}\n")
            if err.strip():
                sys.stdout.write(f"  stderr: {err.strip()}\n")
            continue

        try:
            got = _parse_solver_stdout(out, conf.expect_output_count)
        except Exception as e:
            sys.stdout.write(f"[{i}] FAIL (bad stdout) theta={tc.theta} phi={tc.phi} alpha={tc.alpha}\n")
            sys.stdout.write(f"  err: {e}\n")
            sys.stdout.write(f"  raw stdout: {out!r}\n")
            continue

        exp = [tc.exp_l1, tc.exp_l2, tc.exp_cost]
        maxerr = _max_abs_err(got, exp)
        ok = maxerr <= tc.tol
        if ok:
            earned_points += tc.points
            sys.stdout.write(f"[{i}] PASS  max|err|={maxerr:.3e}\n")
        else:
            sys.stdout.write(f"[{i}] FAIL  max|err|={maxerr:.3e} tol={tc.tol:.1e}\n")
            sys.stdout.write(
                "  got: "
                + " ".join(_fmt_float(x) for x in got)
                + "\n  exp: "
                + " ".join(_fmt_float(x) for x in exp)
                + "\n"
            )

        if worst_case is None or maxerr > worst_case[1]:
            worst_case = (i, maxerr)

    sys.stdout.write(f"\nSCORE: {earned_points:.2f} / {total_points:.2f}\n")
    if worst_case is not None:
        sys.stdout.write(f"WORST: case {worst_case[0]} max|err|={worst_case[1]:.3e}\n")

    # Always exit 0: this is a developer harness meant to show diffs,
    # not to fail the shell on expected intermediate solver failures.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

