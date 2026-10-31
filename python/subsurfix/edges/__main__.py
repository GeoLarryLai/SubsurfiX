# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import sys

from subsurfix.edges import prepare_edges
from subsurfix.errors import (
    ConfigError,
    InvalidInputError,
    MissingDependencyError,
    NotFoundError,
    SubsurfixError,
)

_USAGE = "usage: python -m subsurfix.edges PROJECT_TOML"


def main(argv: list[str] | None = None) -> int:
    """Run ``prepare_edges`` and print each written path."""
    if argv is None:
        argv = sys.argv
    if len(argv) != 2:
        print(_USAGE, file=sys.stderr)
        return 2
    try:
        written = prepare_edges(argv[1])
    except (ConfigError, NotFoundError, InvalidInputError, MissingDependencyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SubsurfixError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in written.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
