# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import argparse
import os
import sys
from enum import Enum

copyright_header = """Copyright (c) Facebook, Inc. and its affiliates. \
All Rights Reserved
SPDX-License-Identifier: MIT"""

lines_header = ["# " + ln + "\n" for ln in copyright_header.splitlines()]


class ReadState(Enum):
    EMPTY = 0
    COMMENT = 1
    TRIPLE_QUOTES = 2


def get_header(filename):
    state = ReadState.EMPTY
    header = []
    with open(filename, "r") as f:
        for line_idx, line in enumerate(f.readlines()):
            line = line.strip()
            # Finite state machine to read "header" of Python source
            # TODO: Can I write this much more compactly with regular expressions?
            if state is ReadState.EMPTY:
                if len(line) and line[0] == "#":
                    state = ReadState.COMMENT
                    header.append(line[1:].strip())
                    continue
                elif len(line) >= 3 and line[:3] == '"""':
                    state = ReadState.TRIPLE_QUOTES
                    header.append(line[3:].strip())
                    continue
                else:
                    # If the file doesn't begin with a comment we consider the
                    # header to be empty
                    return "\n".join(header).strip(), line_idx, state

            elif state is ReadState.COMMENT:
                if len(line) and line[0] == "#":
                    header.append(line[1:].strip())
                    continue
                else:
                    return "\n".join(header).strip(), line_idx, state

            elif state is ReadState.TRIPLE_QUOTES:
                if len(line) >= 3 and '"""' in line:
                    char_idx = line.find('"""')
                    header.append(line[:char_idx].strip())
                    return "\n".join(header).strip(), line_idx, state
                else:
                    header.append(line.strip())
                    continue

            else:
                raise RuntimeError("Invalid read state!")

    # Return error if triple quotes don't terminate
    if state is ReadState.TRIPLE_QUOTES:
        raise RuntimeError(f"Unterminated multi-line string in {f}")

    # If we get to here then the file is all header
    return "\n".join(header).strip(), line_idx + 1, state


def walk_source(paths):
    # Find all Python source files that are not Git ignored
    source_files = set()
    for path in paths:
        for root, _, files in os.walk(path):
            for name in files:
                full_name = os.path.join(root, name)
                if name.endswith(".py") and os.system(
                    f"git check-ignore -q {full_name}"
                ):
                    source_files.add(full_name)

    return sorted(source_files)


def print_results(count_changed, args):
    # Print results
    if count_changed == 0 and args.check:
        print(f"{count_changed} files would be left unchanged.")
    elif count_changed == len(source_files) and args.check:
        print(f"{count_changed} files would be changed.")
    elif args.check:
        print(
            f"""{count_changed} files would be changed and {len(source_files) /
                 - count_changed} files would be unchanged."""
        )
    elif count_changed:
        print(f"{count_changed} files fixed.")


if __name__ == "__main__":
    # Parse command line arguments
    # Example usage: python scripts/copyright_headers.py --check flowtorch tests scripts
    parser = argparse.ArgumentParser(
        description="Checks and adds the Facebook Incubator copyright header"
    )
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="just checks files and does not change any",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="prints extra information on files"
    )
    parser.add_argument(
        "paths", nargs="+", help="paths to search for Python source files"
    )
    args = parser.parse_args()

    source_files = walk_source(args.paths)

    # Loop over source files and get the "header"
    count_changed = 0
    for name in source_files:
        header, line_idx, state = get_header(name)

        # Replace if it's not equal, starts with empty space, or is not a comment
        if (
            header != copyright_header
            or line_idx != 2
            or not state == ReadState.COMMENT
        ):
            count_changed += 1
            if args.verbose:
                print(name)

            if not args.check:
                # Read the file
                with open(name, "r") as f:
                    lines = f.readlines()

                # Replace the header
                # TODO: Debug the following!
                if state == ReadState.TRIPLE_QUOTES:
                    after_quotes = lines[line_idx][
                        (lines[line_idx].find('"""') + 3) :
                    ].lstrip()
                    if after_quotes == "":
                        lines = lines[line_idx + 1 :]
                    elif after_quotes.startswith(";"):
                        lines = [after_quotes[1:].lstrip()] + lines[line_idx + 1 :]
                    else:
                        raise RuntimeError(
                            "Statements must be separated by newlines or semicolons"
                        )
                else:
                    lines = lines[line_idx:]

                lines = lines_header + lines
                filestring = "".join(lines)

                # Save back to disk
                with open(name, "w") as f:
                    f.write(filestring)

    print_results(count_changed, args)
    sys.exit(count_changed)
