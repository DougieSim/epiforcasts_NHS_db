"""
Shared print helpers for ops CLI modules.
"""

from __future__ import annotations


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def passed(msg: str) -> None:
    print(f"[PASS] {msg}")


def check(condition: bool, ok_msg: str, fail_msg: str) -> int:
    """Print pass/fail and return 0 on success, 1 on failure."""
    if condition:
        passed(ok_msg)
        return 0
    fail(fail_msg)
    return 1
