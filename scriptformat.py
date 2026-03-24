#!/usr/bin/env python3
"""
Convert a compact one-liner CLI command into a formatted multi-line
`export CUDA_VISIBLE_DEVICES=0 python ... \` style command.

Usage:
    python format_cmd.py '<your one-liner command>'
    python format_cmd.py  # reads from stdin
    echo '<cmd>' | python format_cmd.py

Notes:
    - Wraps values containing brackets (e.g. [1,4,10]) in single quotes.
    - Flags with no value (e.g. --train_diffusion) are preserved as-is.
    - CUDA_VISIBLE_DEVICES defaults to 0; override with --cuda <N>.
"""

import sys
import re
import argparse


def tokenize_command(cmd: str) -> list[str]:
    """
    Split a raw CLI string into tokens, respecting:
      - single-quoted strings
      - double-quoted strings
      - bracket groups [...]
      - plain whitespace-separated tokens
    """
    tokens = []
    current = []
    i = 0
    while i < len(cmd):
        c = cmd[i]
        # Single-quoted string
        if c == "'":
            j = cmd.index("'", i + 1)
            current.append(cmd[i:j+1])
            i = j + 1
        # Double-quoted string
        elif c == '"':
            j = cmd.index('"', i + 1)
            current.append(cmd[i:j+1])
            i = j + 1
        # Bracket group
        elif c == '[':
            j = cmd.index(']', i)
            current.append(cmd[i:j+1])
            i = j + 1
        # Whitespace: flush current token
        elif c in (' ', '\t', '\n'):
            if current:
                tokens.append(''.join(current))
                current = []
            i += 1
        else:
            current.append(c)
            i += 1
    if current:
        tokens.append(''.join(current))
    return tokens


def needs_quoting(value: str) -> bool:
    """Return True if value should be wrapped in single quotes."""
    return (
        value.startswith('[') or
        ' ' in value or
        ',' in value
    )


def format_command(raw_cmd: str, cuda_device: int = 0, window: int = 0) -> str:
    tokens = tokenize_command(raw_cmd.strip())

    if not tokens:
        raise ValueError("Empty command.")

    # First token must be the script (e.g. main_dmd.py)
    script = tokens[0]
    rest = tokens[1:]

    lines = [f"export CUDA_VISIBLE_DEVICES={cuda_device}", f"# Window {window}"]
    lines.append(f"python {script} \\")

    # Parse flag/value pairs
    i = 0
    entries = []  # list of (flag, value_or_None)
    while i < len(rest):
        tok = rest[i]
        if tok.startswith('--'):
            # Check if next token is a value or another flag (or end)
            if i + 1 < len(rest) and not rest[i + 1].startswith('--'):
                entries.append((tok, rest[i + 1]))
                i += 2
            else:
                entries.append((tok, None))
                i += 1
        else:
            # Positional argument (rare); attach as bare token
            entries.append((tok, None))
            i += 1

    # Build formatted lines
    for idx, (flag, value) in enumerate(entries):
        is_last = (idx == len(entries) - 1)
        cont = "" if is_last else " \\"

        if value is None:
            lines.append(f"  {flag}{cont}")
        else:
            # Quote if needed (strip existing quotes first, then re-quote)
            v = value.strip("'\"")
            if needs_quoting(v):
                v = f"'{v}'"
            lines.append(f"  {flag} {v}{cont}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Format a compact CLI command into a multi-line export style."
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="The one-liner command string (wrap in quotes). "
             "Reads from stdin if omitted.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="CUDA_VISIBLE_DEVICES value (default: 0)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Window value (default: 0)",
    )
    args = parser.parse_args()

    if args.command:
        raw = args.command
    elif not sys.stdin.isatty():
        raw = sys.stdin.read()
    else:
        print("Paste your command below, then press Ctrl+D (Unix) or Ctrl+Z (Windows):")
        raw = sys.stdin.read()

    try:
        result = format_command(raw, cuda_device=args.cuda, window=args.window)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()